#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import decode_weights, decoder_jacobians, encode_weights
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import _load_task_tensors_for_pipeline
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress
from scripts.audit_latent_raw_common_state_rate import (
    AdamState,
    TRUST_SCAN_SCALES,
    _adam_delta,
    _build_projector,
    _cosine,
    _norm,
    _sha256_file,
    _stable_json_hash,
    _validate_adam,
)
from scripts.audit_variant_a_trajectory_realization import (
    _batch_indices,
    _load_cfg,
    _load_start_bank,
    _load_vae,
    _load_weight_pool,
    _loss_acc,
    _run_dir,
    _selected_lr,
    _task_tensor_set,
    _tensor_sha256,
)


METHODS = (
    "raw_adam",
    "latent_adam_native",
    "latent_adam_trust",
    "latent_sgd_jjt_trust",
    "pullback_natural_trust",
    "projected_raw_adam_trust",
)
PRODUCTION_STEPS = 25
PRODUCTION_EVAL_EVERY = 1
PRODUCTION_TRUST_FRACTIONS = (0.25, 1.0)
PRODUCTION_PROJECTOR_RCOND = 1.0e-5
PRODUCTION_START_BANK_SHA256 = "b59f2c50fd389dd78097b30f1c8240c123ba9c66c7fea73c66ee99f4a4cace29"
PRODUCTION_SOURCE_INDICES = (
    15374,
    10003,
    9470,
    1434,
    6026,
    12176,
    942,
    8978,
    4632,
    14718,
    10946,
    13160,
    122,
    14431,
    3402,
    10659,
)
PRODUCTION_LABELS = ("control_seed0", "control_seed1", "control_seed2")
PRODUCTION_CHECKPOINT_SHA256 = (
    "3fbb53ef5cb290b82c4e2a51788f54951544e59e8c78b1d50aff6edfaa0978bb",
    "10e5e0ffbc83b4eeb0993717490f0a469b93d4f4c89d1b48c89641f819b0a0f5",
    "400545a5bf876c36a439b55d0ba63e1a4d641666822933d0b545e51760dfb5ae",
)
PRIMARY_PULLBACK_FRACTION = 1.0
PRIMARY_GAP_RECOVERY_FRACTION = 0.80


def _log(message: str) -> None:
    print(f"[pullback_deployment] {message}", flush=True)


def _task_tensors_fingerprint(task_tensors: dict[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for name, task in sorted(task_tensors.items()):
        payload[str(name)] = {
            "task_name": str(task.task_name),
            "train_images": _tensor_sha256(task.train_images),
            "train_labels": _tensor_sha256(task.train_labels),
            "test_images": _tensor_sha256(task.test_images),
            "test_labels": _tensor_sha256(task.test_labels),
        }
    return _stable_json_hash(payload)


def _expected_eval_steps(steps: int, eval_every: int) -> list[int]:
    values = {0, int(steps)}
    values.update(range(int(eval_every), int(steps) + 1, int(eval_every)))
    return sorted(values)


def _branch_key_set(frame: pd.DataFrame, *, step_column: str) -> set[tuple[str, int, str, float, int]]:
    return {
        (
            str(row.label),
            int(row.source_weight_index),
            str(row.method),
            float(row.trust_fraction),
            int(getattr(row, step_column)),
        )
        for row in frame[
            ["label", "source_weight_index", "method", "trust_fraction", step_column]
        ].itertuples(index=False)
    }


def _first_upward_crossings(
    observations: list[tuple[float, float]],
    target: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        (left, right)
        for left, right in zip(observations, observations[1:], strict=False)
        if math.isfinite(left[1])
        and math.isfinite(right[1])
        and left[1] <= float(target) <= right[1]
        and right[1] > left[1]
    ]


def _context_mismatch_count(frame: pd.DataFrame, start_bank: pd.DataFrame, labels: list[str]) -> int:
    bank = start_bank.copy()
    bank["expected_stream_start_index"] = (
        bank["start_index"].astype(int) if "start_index" in bank.columns else bank["start_bank_position"].astype(int)
    )
    expected = pd.concat([bank.assign(label=str(label)) for label in labels], ignore_index=True, sort=False)
    context = frame[
        ["label", "source_weight_index", "start_bank_position", "stream_start_index", "task_name", "tau"]
    ].drop_duplicates()
    joined = context.merge(
        expected[
            [
                "label",
                "source_weight_index",
                "start_bank_position",
                "expected_stream_start_index",
                "task_name",
                "tau",
            ]
        ],
        on=["label", "source_weight_index"],
        how="left",
        suffixes=("_actual", "_expected"),
        indicator=True,
        validate="many_to_one",
    )
    matched = joined["_merge"].eq("both")
    equal = (
        matched
        & joined["start_bank_position_actual"].eq(joined["start_bank_position_expected"])
        & joined["stream_start_index"].eq(joined["expected_stream_start_index"])
        & joined["task_name_actual"].eq(joined["task_name_expected"])
        & joined["tau_actual"].eq(joined["tau_expected"])
    )
    return int((~equal).sum())


def _propagate_rollout_singularity(diagnostics: pd.DataFrame) -> pd.DataFrame:
    singular_starts = (
        diagnostics.groupby(["label", "source_weight_index"])["singularity_state_local"]
        .any()
        .rename("singularity_stratum")
        .reset_index()
    )
    return diagnostics.merge(
        singular_starts,
        on=["label", "source_weight_index"],
        how="left",
        validate="many_to_one",
    )


def _select_protocol_start_rows(
    start_bank: pd.DataFrame,
    *,
    protocol_mode: str,
    max_starts: int,
) -> pd.DataFrame:
    if protocol_mode == "production":
        if len(start_bank) < len(PRODUCTION_SOURCE_INDICES):
            raise ValueError(
                f"production start bank has {len(start_bank)} rows, required={len(PRODUCTION_SOURCE_INDICES)}"
            )
        return start_bank.iloc[: len(PRODUCTION_SOURCE_INDICES)].copy().reset_index(drop=True)
    if int(max_starts) > 0:
        return start_bank.iloc[: int(max_starts)].copy().reset_index(drop=True)
    return start_bank.copy().reset_index(drop=True)


def _eval(theta: torch.Tensor, *, task_set: Any, spec: Any, tau: float) -> dict[str, float]:
    with torch.no_grad():
        train_loss, train_acc = _loss_acc(theta, task_set=task_set, spec=spec, split="train", tau=tau)
        test_loss, test_acc = _loss_acc(theta, task_set=task_set, spec=spec, split="test", tau=tau)
    return {
        "train_loss": float(train_loss.detach().cpu().item()),
        "train_acc": float(train_acc.detach().cpu().item()),
        "test_loss": float(test_loss.detach().cpu().item()),
        "test_acc": float(test_acc.detach().cpu().item()),
    }


def _theta_grad(
    theta: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    leaf = theta.detach().clone().requires_grad_(True)
    loss, _ = _loss_acc(leaf, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    grad = torch.autograd.grad(loss, leaf)[0].detach()
    return grad, float(loss.detach().cpu().item())


def _latent_grad(
    z: torch.Tensor,
    *,
    vae: torch.nn.Module,
    normalizer: Any,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    leaf = z.detach().clone().requires_grad_(True)
    theta = decode_weights(vae, normalizer, leaf.reshape(1, -1)).squeeze(0)
    loss, _ = _loss_acc(theta, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    grad = torch.autograd.grad(loss, leaf)[0].detach()
    return grad, theta.detach(), float(loss.detach().cpu().item())


def _match_latent_radius(
    *,
    z: torch.Tensor,
    dz: torch.Tensor,
    theta: torch.Tensor,
    target: float,
    vae: torch.nn.Module,
    normalizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, float, float, bool, bool, int, str]:
    observations: list[tuple[float, float]] = [(0.0, 0.0)]
    for scale in TRUST_SCAN_SCALES[1:]:
        with torch.no_grad():
            theta_scan = decode_weights(vae, normalizer, (z + dz * float(scale)).reshape(1, -1)).squeeze(0)
        observations.append((float(scale), _norm(theta_scan - theta)))
    crossings = _first_upward_crossings(observations, float(target))
    bracket_found = bool(crossings)
    if bracket_found:
        (low_scale, low_metric), (high_scale, high_metric) = crossings[0]
        scale = low_scale if abs(low_metric - float(target)) <= abs(high_metric - float(target)) else high_scale
        with torch.no_grad():
            theta_new = decode_weights(vae, normalizer, (z + dz * float(scale)).reshape(1, -1)).squeeze(0)
        radius = _norm(theta_new - theta)
        reason = "scan_endpoint"
        for _ in range(40):
            if abs(radius - float(target)) / max(float(target), 1.0e-30) <= 0.0025:
                reason = "matched"
                break
            mid_scale = 0.5 * (low_scale + high_scale)
            with torch.no_grad():
                theta_mid = decode_weights(vae, normalizer, (z + dz * float(mid_scale)).reshape(1, -1)).squeeze(0)
            mid_radius = _norm(theta_mid - theta)
            if abs(mid_radius - float(target)) < abs(radius - float(target)):
                scale, theta_new, radius = float(mid_scale), theta_mid, float(mid_radius)
            if mid_radius >= float(target):
                high_scale, high_metric = float(mid_scale), float(mid_radius)
            else:
                low_scale, low_metric = float(mid_scale), float(mid_radius)
        else:
            reason = "max_iterations"
    else:
        scale, _ = min(observations, key=lambda item: abs(item[1] - float(target)))
        with torch.no_grad():
            theta_new = decode_weights(vae, normalizer, (z + dz * float(scale)).reshape(1, -1)).squeeze(0)
        radius = _norm(theta_new - theta)
        reason = "no_upward_crossing"
    rel_error = abs(float(radius) - float(target)) / max(float(target), 1.0e-30)
    boundary = bool(scale >= 63.999)
    return (
        (z + dz * float(scale)).detach(),
        theta_new.detach(),
        float(scale),
        float(rel_error),
        boundary,
        bracket_found,
        int(len(crossings)),
        reason,
    )


def _executed_preimage_residuals(
    projector: Any,
    vector: torch.Tensor,
    dz_executed: torch.Tensor,
    *,
    scale: float,
) -> dict[str, float]:
    target64 = projector.project64(vector) * float(scale)
    ideal64 = projector.latent_preimage64(vector) * float(scale)
    executed64 = dz_executed.detach().double()
    denominator = max(_norm(target64), 1.0e-30)
    algebraic = _norm(projector.jacobian64 @ ideal64 - target64) / denominator
    quantization = _norm(projector.jacobian64 @ (executed64 - ideal64)) / denominator
    executable = _norm(projector.jacobian64 @ executed64 - target64) / denominator
    return {
        "algebraic": float(algebraic),
        "quantization": float(quantization),
        "executable": float(executable),
    }


def _is_retraction_executable(method: str, relative_error: float, cosine: float) -> bool:
    if method not in {"latent_sgd_jjt_trust", "pullback_natural_trust", "projected_raw_adam_trust"}:
        return True
    return bool(
        math.isfinite(float(relative_error))
        and math.isfinite(float(cosine))
        and float(relative_error) <= 0.25
        and float(cosine) >= 0.95
    )


def _raw_reference(
    *,
    theta0: torch.Tensor,
    raw_lr: float,
    steps: int,
    eval_every: int,
    task_set: Any,
    spec: Any,
    tau: float,
    stream_start_index: int,
    batch_size: int,
    common: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]]]:
    value = theta0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([value], lr=float(raw_lr))
    rows: list[dict[str, Any]] = []
    radii: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    cumulative = 0.0
    for step in range(int(steps) + 1):
        if step == 0 or step % int(eval_every) == 0 or step == int(steps):
            rows.append(
                {
                    **common,
                    "method": "raw_adam",
                    "trust_fraction": 1.0,
                    "step": int(step),
                    "cumulative_weight_path": cumulative,
                    "state_theta_sha256": _tensor_sha256(value.detach()),
                    **_eval(value.detach(), task_set=task_set, spec=spec, tau=tau),
                }
            )
        if step == int(steps):
            break
        batch_indices = _batch_indices(task_set, batch_size=batch_size, step=step, start_bank_position=stream_start_index)
        before = value.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss_acc(value, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
        loss.backward()
        optimizer.step()
        delta = value.detach() - before
        radius = _norm(delta)
        radii.append(radius)
        cumulative += radius
        diagnostics.append(
            {
                **common,
                "method": "raw_adam",
                "trust_fraction": 1.0,
                "proposal_step": int(step),
                "batch_indices_sha256": _tensor_sha256(batch_indices),
                "batch_loss": float(loss.detach().cpu().item()),
                "target_raw_radius": radius,
                "realized_radius": radius,
                "trust_match_rel_error": 0.0,
                "trust_scale": 1.0,
                "trust_scale_boundary_hit": False,
                "trust_bracket_found": True,
                "trust_crossing_count": 1,
                "trust_termination_reason": "raw_reference",
                "projector_rank": -1,
                "projector_condition": float("nan"),
                "retraction_rel_error": 0.0,
                "retraction_cosine": 1.0,
                "operator_executable": True,
                "retraction_executable": True,
                "trust_executable": True,
                "treatment_executable": True,
                "singularity_state_local": False,
            }
        )
    return rows, radii, diagnostics


def _latent_curve(
    *,
    method: str,
    trust_fraction: float,
    z0: torch.Tensor,
    target_radii: list[float],
    latent_lr: float,
    steps: int,
    eval_every: int,
    task_set: Any,
    spec: Any,
    tau: float,
    stream_start_index: int,
    batch_size: int,
    vae: torch.nn.Module,
    normalizer: Any,
    projector_rcond: float,
    jacobian_chunk_size: int,
    common: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    z = z0.detach().clone()
    latent_state = AdamState.zeros_like(z)
    raw_shadow_state: AdamState | None = None
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    cumulative = 0.0
    for step in range(int(steps) + 1):
        with torch.no_grad():
            theta = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).detach()
        if step == 0 or step % int(eval_every) == 0 or step == int(steps):
            rows.append(
                {
                    **common,
                    "method": method,
                    "trust_fraction": float(trust_fraction),
                    "step": int(step),
                    "cumulative_weight_path": cumulative,
                    "state_theta_sha256": _tensor_sha256(theta),
                    **_eval(theta, task_set=task_set, spec=spec, tau=tau),
                }
            )
        if step == int(steps):
            break
        target_raw_radius = float(target_radii[step])
        target_radius = target_raw_radius * float(trust_fraction)
        batch_indices = _batch_indices(task_set, batch_size=batch_size, step=step, start_bank_position=stream_start_index)
        projector_rank = -1
        projector_condition = float("nan")
        projector_rank_rcond_1e6 = -1
        projector_rank_rcond_1e5 = -1
        projector_rank_rcond_1e4 = -1
        projector_condition_rcond_1e6 = float("nan")
        tangent_fraction = float("nan")
        jacobian: torch.Tensor | None = None
        projector: Any | None = None
        ambient_proposal: torch.Tensor | None = None
        linear_delta: torch.Tensor | None = None
        preimage_algebraic_residual = float("nan")
        preimage_quantization_residual = float("nan")
        preimage_executable_residual = float("nan")
        executed_preimage_algebraic_residual = float("nan")
        executed_preimage_quantization_residual = float("nan")
        executed_preimage_executable_residual = float("nan")
        bracket_found = True
        crossing_count = 1
        match_reason = "native"
        if method in {"latent_adam_native", "latent_adam_trust"}:
            grad_z, _theta_check, batch_loss = _latent_grad(
                z,
                vae=vae,
                normalizer=normalizer,
                task_set=task_set,
                spec=spec,
                tau=tau,
                batch_indices=batch_indices,
            )
            dz = _adam_delta(latent_state, grad_z, lr=float(latent_lr))
            if method == "latent_adam_native":
                with torch.no_grad():
                    theta_new = decode_weights(vae, normalizer, (z + dz).reshape(1, -1)).squeeze(0).detach()
                z_new = (z + dz).detach()
                scale = 1.0
                match_error = abs(_norm(theta_new - theta) - target_radius) / max(target_radius, 1.0e-30)
                boundary = False
            else:
                z_new, theta_new, scale, match_error, boundary, bracket_found, crossing_count, match_reason = _match_latent_radius(
                    z=z,
                    dz=dz,
                    theta=theta,
                    target=target_radius,
                    vae=vae,
                    normalizer=normalizer,
                )
        elif method in {"latent_sgd_jjt_trust", "pullback_natural_trust", "projected_raw_adam_trust"}:
            grad_theta, batch_loss = _theta_grad(
                theta,
                task_set=task_set,
                spec=spec,
                tau=tau,
                batch_indices=batch_indices,
            )
            jacobian = decoder_jacobians(
                vae,
                normalizer,
                None,
                z.reshape(1, -1),
                create_graph=False,
                chunk_size=int(jacobian_chunk_size),
            ).squeeze(0).detach()
            if method != "latent_sgd_jjt_trust":
                projector = _build_projector(jacobian, rcond=float(projector_rcond))
                projector_rank = int(projector.rank)
                kept = projector.singular_values[projector.keep]
                projector_condition = float(kept.max().cpu().item() / kept.min().cpu().item()) if kept.numel() else float("inf")
                singular_values = projector.singular_values
                singular_max = float(singular_values.max().cpu().item())
                ranks = {
                    rcond: int((singular_values > singular_max * rcond).sum().cpu().item())
                    for rcond in (1.0e-6, 1.0e-5, 1.0e-4)
                }
                projector_rank_rcond_1e6 = ranks[1.0e-6]
                projector_rank_rcond_1e5 = ranks[1.0e-5]
                projector_rank_rcond_1e4 = ranks[1.0e-4]
                kept_1e6 = singular_values[singular_values > singular_max * 1.0e-6]
                projector_condition_rcond_1e6 = (
                    float(kept_1e6.max().cpu().item() / kept_1e6.min().cpu().item())
                    if kept_1e6.numel()
                    else float("inf")
                )
            if method == "latent_sgd_jjt_trust":
                ambient_proposal = -grad_theta
                tangent = jacobian @ (jacobian.T @ ambient_proposal)
                tangent_fraction = _norm(tangent) / max(_norm(ambient_proposal), 1.0e-30)
                dz = jacobian.T @ ambient_proposal
            elif method == "pullback_natural_trust":
                ambient_proposal = -grad_theta
            else:
                if raw_shadow_state is None:
                    raw_shadow_state = AdamState.zeros_like(grad_theta)
                ambient_proposal = _adam_delta(raw_shadow_state, grad_theta, lr=float(common["raw_lr"]))
            if projector is not None:
                tangent = projector.project(ambient_proposal)
                tangent_fraction = _norm(tangent) / max(_norm(ambient_proposal), 1.0e-30)
                dz = projector.latent_preimage(ambient_proposal)
                residuals = projector.preimage_residuals(ambient_proposal, dz)
                preimage_algebraic_residual = residuals["algebraic"]
                preimage_quantization_residual = residuals["quantization"]
                preimage_executable_residual = residuals["executable"]
            linear_delta = tangent
            z_new, theta_new, scale, match_error, boundary, bracket_found, crossing_count, match_reason = _match_latent_radius(
                z=z,
                dz=dz,
                theta=theta,
                target=target_radius,
                vae=vae,
                normalizer=normalizer,
            )
        else:
            raise ValueError(f"unknown method={method!r}")
        delta = theta_new - theta
        dz_executed = z_new - z
        realized_radius = _norm(delta)
        cumulative += realized_radius
        retraction = float("nan")
        linear_cos = float("nan")
        proposal_retraction = float("nan")
        proposal_linear_cos = float("nan")
        if linear_delta is not None and jacobian is not None:
            scaled_linear = linear_delta * float(scale)
            proposal_retraction = _norm(delta - scaled_linear) / max(_norm(scaled_linear), 1.0e-30)
            proposal_linear_cos = _cosine(delta, scaled_linear)
            executed_linear = jacobian @ dz_executed
            retraction = _norm(delta - executed_linear) / max(_norm(executed_linear), 1.0e-30)
            linear_cos = _cosine(delta, executed_linear)
        if projector is not None and ambient_proposal is not None:
            executed_residuals = _executed_preimage_residuals(
                projector,
                ambient_proposal,
                dz_executed,
                scale=float(scale),
            )
            executed_preimage_algebraic_residual = executed_residuals["algebraic"]
            executed_preimage_quantization_residual = executed_residuals["quantization"]
            executed_preimage_executable_residual = executed_residuals["executable"]
        trust_executable = bool(
            method == "latent_adam_native"
            or (
                bracket_found
                and not boundary
                and math.isfinite(float(match_error))
                and float(match_error) <= 0.01
            )
        )
        operator_executable = bool(
            projector is None
            or (
                executed_preimage_algebraic_residual <= 1.0e-10
                and executed_preimage_executable_residual <= 1.0e-4
                and executed_preimage_executable_residual
                <= 1.25
                * (executed_preimage_algebraic_residual + executed_preimage_quantization_residual)
                + 1.0e-7
            )
        )
        retraction_executable = _is_retraction_executable(method, retraction, linear_cos)
        singularity_stratum = bool(
            projector is not None
            and (
                projector_condition_rcond_1e6 > 1.0e5
                or projector_rank_rcond_1e6 != projector_rank_rcond_1e5
            )
        )
        diagnostics.append(
            {
                **common,
                "method": method,
                "trust_fraction": float(trust_fraction),
                "proposal_step": int(step),
                "batch_indices_sha256": _tensor_sha256(batch_indices),
                "batch_loss": float(batch_loss),
                "target_raw_radius": target_raw_radius,
                "target_trust_radius": target_radius,
                "realized_radius": realized_radius,
                "trust_match_rel_error": float(match_error),
                "trust_scale": float(scale),
                "trust_scale_boundary_hit": bool(boundary),
                "trust_bracket_found": bool(bracket_found),
                "trust_crossing_count": int(crossing_count),
                "trust_termination_reason": str(match_reason),
                "projector_rank": projector_rank,
                "projector_condition": projector_condition,
                "projector_condition_rcond_1e6": projector_condition_rcond_1e6,
                "projector_rank_rcond_1e6": projector_rank_rcond_1e6,
                "projector_rank_rcond_1e5": projector_rank_rcond_1e5,
                "projector_rank_rcond_1e4": projector_rank_rcond_1e4,
                "ambient_tangent_norm_fraction": tangent_fraction,
                "retraction_rel_error": retraction,
                "retraction_cosine": linear_cos,
                "proposal_retraction_rel_error": proposal_retraction,
                "proposal_retraction_cosine": proposal_linear_cos,
                "preimage_algebraic_residual": preimage_algebraic_residual,
                "preimage_quantization_residual": preimage_quantization_residual,
                "preimage_executable_residual": preimage_executable_residual,
                "executed_preimage_algebraic_residual": executed_preimage_algebraic_residual,
                "executed_preimage_quantization_residual": executed_preimage_quantization_residual,
                "executed_preimage_executable_residual": executed_preimage_executable_residual,
                "trust_executable": trust_executable,
                "operator_executable": operator_executable,
                "retraction_executable": retraction_executable,
                "treatment_executable": bool(
                    trust_executable and operator_executable and retraction_executable
                ),
                "singularity_state_local": singularity_stratum,
            }
        )
        z = z_new
    return rows, diagnostics


def _summary(curves: pd.DataFrame, diagnostics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in curves.groupby(["label", "source_weight_index", "method", "trust_fraction"]):
        label, source, method, trust_fraction = keys
        group = group.sort_values("step")
        base = group.iloc[0]
        post = group[group["step"] > 0]
        steps = post["step"].to_numpy(dtype=np.float64)
        for metric in ("train_loss", "test_loss"):
            values = post[metric].to_numpy(dtype=np.float64)
            slope = float(np.polyfit(steps, values, 1)[0]) if len(values) >= 2 else float("nan")
            rows.append(
                {
                    "label": label,
                    "source_weight_index": int(source),
                    "method": method,
                    "trust_fraction": float(trust_fraction),
                    "metric": metric,
                    "step0": float(base[metric]),
                    "post0_mean": float(np.mean(values)),
                    "post0_progress": float(base[metric] - np.mean(values)),
                    "final": float(values[-1]),
                    "final_progress": float(base[metric] - values[-1]),
                    "ols_slope": slope,
                }
            )
    per_start = pd.DataFrame(rows)
    trajectory_flags = (
        diagnostics.groupby(["label", "source_weight_index", "method", "trust_fraction"], as_index=False)
        .agg(
            trajectory_trust_executable=("trust_executable", "all"),
            trajectory_operator_executable=("operator_executable", "all"),
            trajectory_retraction_executable=("retraction_executable", "all"),
            trajectory_treatment_executable=("treatment_executable", "all"),
            trajectory_singularity_stratum=("singularity_stratum", "any"),
        )
    )
    per_start = per_start.merge(
        trajectory_flags,
        on=["label", "source_weight_index", "method", "trust_fraction"],
        how="left",
        validate="many_to_one",
    )
    per_start.to_csv(out_dir / "pullback_per_start_summary.csv", index=False)
    aggregate = (
        per_start.groupby(["label", "method", "trust_fraction", "metric"])[["post0_progress", "final_progress", "ols_slope"]]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    aggregate.columns = ["_".join(str(value) for value in column if str(value)) for column in aggregate.columns]
    aggregate.to_csv(out_dir / "pullback_method_summary.csv", index=False)
    return per_start


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, reps: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, values.size, size=(int(reps), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_contrasts(per_start: pd.DataFrame, out_dir: Path, trust_fractions: list[float]) -> pd.DataFrame:
    specifications: list[tuple[str, str, float, str, float]] = [
        ("latent_native_minus_raw", "latent_adam_native", 1.0, "raw_adam", 1.0),
    ]
    for fraction in trust_fractions:
        specifications.extend(
            [
                (f"latent_trust_f{fraction:g}_minus_raw", "latent_adam_trust", fraction, "raw_adam", 1.0),
                (
                    f"jjt_f{fraction:g}_minus_latent_trust",
                    "latent_sgd_jjt_trust",
                    fraction,
                    "latent_adam_trust",
                    fraction,
                ),
                (
                    f"pullback_f{fraction:g}_minus_latent_trust",
                    "pullback_natural_trust",
                    fraction,
                    "latent_adam_trust",
                    fraction,
                ),
                (
                    f"pullback_f{fraction:g}_minus_jjt",
                    "pullback_natural_trust",
                    fraction,
                    "latent_sgd_jjt_trust",
                    fraction,
                ),
                (f"pullback_f{fraction:g}_minus_raw", "pullback_natural_trust", fraction, "raw_adam", 1.0),
                (
                    f"projected_raw_f{fraction:g}_minus_raw",
                    "projected_raw_adam_trust",
                    fraction,
                    "raw_adam",
                    1.0,
                ),
                (
                    f"projected_raw_f{fraction:g}_minus_pullback",
                    "projected_raw_adam_trust",
                    fraction,
                    "pullback_natural_trust",
                    fraction,
                ),
                (
                    f"projected_raw_f{fraction:g}_minus_jjt",
                    "projected_raw_adam_trust",
                    fraction,
                    "latent_sgd_jjt_trust",
                    fraction,
                ),
                (
                    f"projected_raw_f{fraction:g}_minus_latent_trust",
                    "projected_raw_adam_trust",
                    fraction,
                    "latent_adam_trust",
                    fraction,
                ),
            ]
        )
    per_start_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for contrast, left_method, left_fraction, right_method, right_fraction in specifications:
        for metric in ("train_loss", "test_loss"):
            left = per_start[
                (per_start["method"] == left_method)
                & np.isclose(per_start["trust_fraction"].astype(float), float(left_fraction))
                & (per_start["metric"] == metric)
            ]
            right = per_start[
                (per_start["method"] == right_method)
                & np.isclose(per_start["trust_fraction"].astype(float), float(right_fraction))
                & (per_start["metric"] == metric)
            ]
            joined = left.merge(
                right,
                on=["label", "source_weight_index", "metric"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            joined["pair_executable"] = (
                joined["trajectory_treatment_executable_left"].astype(bool)
                & joined["trajectory_treatment_executable_right"].astype(bool)
            )
            joined["pair_regular_executable"] = (
                joined["pair_executable"]
                & ~joined["trajectory_singularity_stratum_left"].astype(bool)
                & ~joined["trajectory_singularity_stratum_right"].astype(bool)
            )
            for score, sign in (("post0_progress", 1.0), ("final_progress", 1.0), ("ols_slope", -1.0)):
                joined = joined.copy()
                joined["contrast_value"] = sign * (joined[f"{score}_left"] - joined[f"{score}_right"])
                for row in joined.itertuples(index=False):
                    per_start_rows.append(
                        {
                            "contrast": contrast,
                            "label": row.label,
                            "source_weight_index": int(row.source_weight_index),
                            "metric": metric,
                            "score": score,
                            "endpoint_role": "primary" if metric == "train_loss" and score == "post0_progress" else "secondary",
                            "left": left_method,
                            "left_trust_fraction": float(left_fraction),
                            "right": right_method,
                            "right_trust_fraction": float(right_fraction),
                            "pair_executable": bool(row.pair_executable),
                            "pair_regular_executable": bool(row.pair_regular_executable),
                            "contrast_value_left_better_positive": float(row.contrast_value),
                        }
                    )
                for analysis_set, eligible in (
                    ("all_start_deployment_itt", np.ones(len(joined), dtype=bool)),
                    ("executable_sensitivity", joined["pair_executable"].to_numpy(dtype=bool)),
                    ("regular_executable_sensitivity", joined["pair_regular_executable"].to_numpy(dtype=bool)),
                ):
                    eligible_rows = joined.loc[eligible]
                    for label, label_group in eligible_rows.groupby("label"):
                        values = label_group["contrast_value"].to_numpy(dtype=np.float64)
                        summary_rows.append(
                            {
                                "inference_scope": "within_vae_seed_diagnostic",
                                "analysis_set": analysis_set,
                                "contrast": contrast,
                                "label": str(label),
                                "metric": metric,
                                "score": score,
                                "endpoint_role": "primary" if metric == "train_loss" and score == "post0_progress" else "secondary",
                                "left": left_method,
                                "left_trust_fraction": float(left_fraction),
                                "right": right_method,
                                "right_trust_fraction": float(right_fraction),
                                "n_starts": int(values.size),
                                "n_vae_seeds": 1,
                                "mean_left_better_positive": float(np.mean(values)),
                                "median_left_better_positive": float(np.median(values)),
                                "bootstrap95_low": float("nan"),
                                "bootstrap95_high": float("nan"),
                                "left_better_count": int(np.sum(values > 0.0)),
                                "left_worse_count": int(np.sum(values < 0.0)),
                            }
                        )
                    seed_means = eligible_rows.groupby("label")["contrast_value"].mean().to_numpy(dtype=np.float64)
                    if seed_means.size:
                        lo, hi = (
                            _bootstrap_mean_ci(seed_means, seed=8191 + len(summary_rows))
                            if seed_means.size >= 3
                            else (float("nan"), float("nan"))
                        )
                        summary_rows.append(
                            {
                                "inference_scope": "vae_seed_clustered",
                                "analysis_set": analysis_set,
                                "contrast": contrast,
                                "label": "__across_vae_seeds__",
                                "metric": metric,
                                "score": score,
                                "endpoint_role": "primary" if metric == "train_loss" and score == "post0_progress" else "secondary",
                                "left": left_method,
                                "left_trust_fraction": float(left_fraction),
                                "right": right_method,
                                "right_trust_fraction": float(right_fraction),
                                "n_starts": int(eligible_rows["source_weight_index"].nunique()),
                                "n_vae_seeds": int(seed_means.size),
                                "mean_left_better_positive": float(np.mean(seed_means)),
                                "median_left_better_positive": float(np.median(seed_means)),
                                "bootstrap95_low": lo,
                                "bootstrap95_high": hi,
                                "left_better_count": int(np.sum(seed_means > 0.0)),
                                "left_worse_count": int(np.sum(seed_means < 0.0)),
                            }
                        )
    pd.DataFrame(per_start_rows).to_csv(out_dir / "pullback_paired_per_start.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "pullback_paired_summary.csv", index=False)
    return summary


def _gap_closure_summary(per_start: pd.DataFrame, out_dir: Path, *, production_mode: bool) -> pd.DataFrame:
    primary = per_start[
        (per_start["metric"] == "train_loss")
        & per_start["method"].isin(
            ["raw_adam", "latent_adam_trust", "latent_sgd_jjt_trust", "pullback_natural_trust"]
        )
        & (
            (per_start["method"] == "raw_adam")
            | np.isclose(per_start["trust_fraction"].astype(float), PRIMARY_PULLBACK_FRACTION)
        )
    ]
    seed_means = (
        primary.groupby(["label", "method"], as_index=False)["post0_progress"]
        .mean()
        .pivot(index="label", columns="method", values="post0_progress")
        .reset_index()
    )
    required = ["raw_adam", "latent_adam_trust", "latent_sgd_jjt_trust", "pullback_natural_trust"]
    missing = [column for column in required if column not in seed_means.columns]
    if missing:
        raise RuntimeError(f"gap-closure summary is missing methods: {missing}")
    rows: list[dict[str, Any]] = []
    for row in seed_means.itertuples(index=False):
        raw = float(row.raw_adam)
        matched = float(row.latent_adam_trust)
        jjt = float(row.latent_sgd_jjt_trust)
        pullback = float(row.pullback_natural_trust)
        gap = raw - matched
        recovered = pullback - matched
        recovery = recovered / gap if gap > 0.0 else float("nan")
        rows.append(
            {
                "scope": "vae_seed",
                "label": str(row.label),
                "primary_metric": "train_post0_aulc_progress",
                "primary_fraction": PRIMARY_PULLBACK_FRACTION,
                "raw_progress": raw,
                "matched_latent_progress": matched,
                "jjt_progress": jjt,
                "pullback_progress": pullback,
                "raw_minus_matched_gap": gap,
                "pullback_minus_matched": recovered,
                "pullback_minus_jjt": pullback - jjt,
                "raw_minus_pullback_residual": raw - pullback,
                "gap_recovery_fraction": recovery,
                "gap_exists": bool(gap > 0.0),
                "operator_direction_pass": bool(pullback > matched and pullback > jjt),
                "recovery_80pct_pass": bool(gap > 0.0 and recovered >= PRIMARY_GAP_RECOVERY_FRACTION * gap),
                "production_success_eligible": False,
            }
        )
    seed_rows = pd.DataFrame(rows)
    aggregate_values = seed_means[required].mean()
    raw = float(aggregate_values["raw_adam"])
    matched = float(aggregate_values["latent_adam_trust"])
    jjt = float(aggregate_values["latent_sgd_jjt_trust"])
    pullback = float(aggregate_values["pullback_natural_trust"])
    gap = raw - matched
    recovered = pullback - matched
    recovery = recovered / gap if gap > 0.0 else float("nan")
    direction_seed_count = int(seed_rows["operator_direction_pass"].sum())
    recovery_seed_count = int(seed_rows["recovery_80pct_pass"].sum())
    production_eligible = bool(production_mode and len(seed_rows) == len(PRODUCTION_LABELS))
    success = bool(
        production_eligible
        and gap > 0.0
        and pullback > matched
        and pullback > jjt
        and recovery >= PRIMARY_GAP_RECOVERY_FRACTION
        and direction_seed_count == len(PRODUCTION_LABELS)
    )
    aggregate = pd.DataFrame(
        [
            {
                "scope": "across_vae_seed_means",
                "label": "__across_vae_seeds__",
                "primary_metric": "train_post0_aulc_progress",
                "primary_fraction": PRIMARY_PULLBACK_FRACTION,
                "raw_progress": raw,
                "matched_latent_progress": matched,
                "jjt_progress": jjt,
                "pullback_progress": pullback,
                "raw_minus_matched_gap": gap,
                "pullback_minus_matched": recovered,
                "pullback_minus_jjt": pullback - jjt,
                "raw_minus_pullback_residual": raw - pullback,
                "gap_recovery_fraction": recovery,
                "gap_exists": bool(gap > 0.0),
                "operator_direction_pass": bool(pullback > matched and pullback > jjt),
                "recovery_80pct_pass": bool(gap > 0.0 and recovery >= PRIMARY_GAP_RECOVERY_FRACTION),
                "direction_pass_seed_count": direction_seed_count,
                "recovery_pass_seed_count": recovery_seed_count,
                "n_vae_seeds": int(len(seed_rows)),
                "production_success_eligible": production_eligible,
                "predeclared_rc5c_sufficiency_success": success,
            }
        ]
    )
    output = pd.concat([seed_rows, aggregate], ignore_index=True, sort=False)
    output.to_csv(out_dir / "pullback_gap_closure.csv", index=False)
    return output


def _plot_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    labels = sorted(curves["label"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        len(labels),
        2,
        figsize=(14, max(4.5, 4.2 * len(labels))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, label in enumerate(labels):
        label_rows = curves[curves["label"].astype(str) == label]
        for (method, trust_fraction), group in label_rows.groupby(["method", "trust_fraction"]):
            means = group.groupby("step")[["train_loss", "test_loss"]].mean().reset_index()
            display = f"{method} f={float(trust_fraction):g}"
            axes[row_index, 0].plot(means["step"], means["train_loss"], label=display)
            axes[row_index, 1].plot(means["step"], means["test_loss"], label=display)
        axes[row_index, 0].set_title(f"{label}: full-train loss")
        axes[row_index, 1].set_title(f"{label}: test loss")
        for axis in axes[row_index]:
            axis.set_xlabel("optimizer step")
            axis.set_ylabel("cross entropy")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    fig.savefig(out_dir / "pullback_curves.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")
    if len(set(str(value) for value in args.label)) != len(args.label):
        raise ValueError("--label values must be unique VAE training-seed identifiers")
    run_dirs = [_run_dir(value) for value in args.run]
    labels = [str(value) for value in args.label]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 or float(value) > 1.0 for value in args.trust_fraction):
        raise ValueError(f"--trust-fraction values must be in (0,1], got={args.trust_fraction}")
    if int(args.steps) <= 0 or int(args.eval_every) <= 0:
        raise ValueError("--steps and --eval-every must be positive")
    source_indices = [int(value) for value in args.source_index]
    if args.protocol_mode == "production" and (source_indices or int(args.max_starts) != 0):
        raise ValueError("production protocol forbids --source-index and --max-starts")
    _log(
        f"startup runs={[str(v) for v in run_dirs]} labels={labels} device={args.device} dtype=float32 seed={args.seed} "
        f"steps={args.steps} eval_every={args.eval_every} projector_rcond={args.projector_rcond} "
        f"jacobian_chunk_size={args.jacobian_chunk_size} output_dir={out_dir}"
    )
    base_cfg = _load_cfg(run_dirs[0], device=str(args.device), downstream_steps=int(args.steps))
    device = torch.device(base_cfg.device)
    dtype = torch_dtype(base_cfg)
    adam_error = _validate_adam(device)
    task_tensors = _load_task_tensors_for_pipeline(base_cfg, device=device, dtype=dtype)
    task_fingerprints = [_task_tensors_fingerprint(task_tensors)]
    fairness_batch_sizes = [int(base_cfg.downstream_batch_size)]
    for run_dir in run_dirs[1:]:
        other_cfg = _load_cfg(run_dir, device=str(args.device), downstream_steps=int(args.steps))
        other_tasks = _load_task_tensors_for_pipeline(other_cfg, device=device, dtype=dtype)
        task_fingerprints.append(_task_tensors_fingerprint(other_tasks))
        fairness_batch_sizes.append(int(other_cfg.downstream_batch_size))
    if len(set(task_fingerprints)) != 1:
        raise RuntimeError(f"task tensor mismatch across VAE seeds: {task_fingerprints}")
    if len(set(fairness_batch_sizes)) != 1:
        raise RuntimeError(f"downstream batch-size mismatch across VAE seeds: {fairness_batch_sizes}")
    weight_pool_file_hashes = [_sha256_file(run_dir / "weight_pool.pt") for run_dir in run_dirs]
    weight_record_file_hashes = [_sha256_file(run_dir / "weight_pool_records.csv") for run_dir in run_dirs]
    checkpoint_file_hashes = [_sha256_file(run_dir / "vae_checkpoint.pt") for run_dir in run_dirs]
    if len(set(weight_pool_file_hashes)) != 1 or len(set(weight_record_file_hashes)) != 1:
        raise RuntimeError(
            f"weight-pool provenance mismatch across VAE seeds: pools={weight_pool_file_hashes} records={weight_record_file_hashes}"
        )
    resolved_lrs = [
        {
            "label": label,
            "raw_lr": float(_selected_lr(run_dir, "raw")),
            "latent_lr": float(_selected_lr(run_dir, "decoder_latent")),
        }
        for run_dir, label in zip(run_dirs, labels, strict=True)
    ]
    if len({row["raw_lr"] for row in resolved_lrs}) != 1 or len({row["latent_lr"] for row in resolved_lrs}) != 1:
        raise RuntimeError(f"selected LR mismatch across VAE seeds: {resolved_lrs}")
    weights_cpu, _records, weight_key, spec = _load_weight_pool(run_dirs[0])
    weights_device = weights_cpu.to(device=device, dtype=dtype)
    start_bank = _load_start_bank(Path(args.start_bank_csv).expanduser().resolve(), source_indices)
    start_bank = _select_protocol_start_rows(
        start_bank,
        protocol_mode=str(args.protocol_mode),
        max_starts=int(args.max_starts),
    )
    if args.protocol_mode == "production":
        violations: list[str] = []
        if int(args.steps) != PRODUCTION_STEPS:
            violations.append(f"steps={args.steps}")
        if int(args.eval_every) != PRODUCTION_EVAL_EVERY:
            violations.append(f"eval_every={args.eval_every}")
        if tuple(float(value) for value in args.trust_fraction) != PRODUCTION_TRUST_FRACTIONS:
            violations.append(f"trust_fractions={args.trust_fraction}")
        if not math.isclose(float(args.projector_rcond), PRODUCTION_PROJECTOR_RCOND, rel_tol=0.0, abs_tol=1.0e-15):
            violations.append(f"projector_rcond={args.projector_rcond}")
        source_order = tuple(start_bank["source_weight_index"].astype(int).tolist())
        if source_order != PRODUCTION_SOURCE_INDICES:
            violations.append(f"ordered production start bank mismatch, got={source_order}")
        start_bank_hash = _sha256_file(Path(args.start_bank_csv).expanduser().resolve())
        if start_bank_hash != PRODUCTION_START_BANK_SHA256:
            violations.append(f"start_bank_sha256={start_bank_hash}")
        if tuple(labels) != PRODUCTION_LABELS:
            violations.append(f"labels={labels}")
        if tuple(checkpoint_file_hashes) != PRODUCTION_CHECKPOINT_SHA256:
            violations.append(f"checkpoint_hashes={checkpoint_file_hashes}")
        if violations:
            raise ValueError("production protocol mismatch: " + "; ".join(violations))
    start_bank.to_csv(out_dir / "pullback_start_bank.csv", index=False)
    all_curves: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    lr_rows: list[dict[str, Any]] = []
    for run_dir, label in zip(run_dirs, labels, strict=True):
        cfg = _load_cfg(run_dir, device=str(args.device), downstream_steps=int(args.steps))
        run_weights, _run_records, run_key, run_spec = _load_weight_pool(run_dir)
        if run_key != weight_key or tuple(run_weights.shape) != tuple(weights_cpu.shape) or int(run_spec.dim) != int(spec.dim):
            raise RuntimeError(f"weight pool mismatch for {run_dir}")
        vae, normalizer, _checkpoint = _load_vae(run_dir, cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
        raw_lr = _selected_lr(run_dir, "raw")
        latent_lr = _selected_lr(run_dir, "decoder_latent")
        lr_rows.append({"label": label, "raw_lr": raw_lr, "latent_lr": latent_lr, "run_dir": str(run_dir)})
        progress = make_progress(cfg, total=len(start_bank), desc=f"pullback deployment {label}")
        label_curves: list[dict[str, Any]] = []
        label_diagnostics: list[dict[str, Any]] = []
        try:
            for _, start_row in start_bank.iterrows():
                source = int(start_row["source_weight_index"])
                position = int(start_row["start_bank_position"])
                stream_index = int(start_row.get("start_index", position))
                task_name = str(start_row["task_name"])
                tau = float(start_row["tau"])
                task_set = _task_tensor_set(task_tensors, task_name)
                w0 = weights_device[source].detach()
                with torch.no_grad():
                    z0 = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
                    theta0 = decode_weights(vae, normalizer, z0.reshape(1, -1)).squeeze(0).detach()
                common = {
                    "label": label,
                    "run_name": run_dir.name,
                    "source_weight_index": source,
                    "start_bank_position": position,
                    "stream_start_index": stream_index,
                    "task_name": task_name,
                    "tau": tau,
                    "raw_lr": raw_lr,
                    "latent_lr": latent_lr,
                    "theta0_sha256": _tensor_sha256(theta0),
                }
                raw_curves, radii, raw_diagnostics = _raw_reference(
                    theta0=theta0,
                    raw_lr=raw_lr,
                    steps=int(args.steps),
                    eval_every=int(args.eval_every),
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    stream_start_index=stream_index,
                    batch_size=int(cfg.downstream_batch_size),
                    common=common,
                )
                label_curves.extend(raw_curves)
                label_diagnostics.extend(raw_diagnostics)
                for method in METHODS[1:]:
                    fractions = (1.0,) if method == "latent_adam_native" else tuple(args.trust_fraction)
                    for trust_fraction in fractions:
                        curves, diagnostics = _latent_curve(
                            method=method,
                            trust_fraction=float(trust_fraction),
                            z0=z0,
                            target_radii=radii,
                            latent_lr=latent_lr,
                            steps=int(args.steps),
                            eval_every=int(args.eval_every),
                            task_set=task_set,
                            spec=spec,
                            tau=tau,
                            stream_start_index=stream_index,
                            batch_size=int(cfg.downstream_batch_size),
                            vae=vae,
                            normalizer=normalizer,
                            projector_rcond=float(args.projector_rcond),
                            jacobian_chunk_size=int(args.jacobian_chunk_size),
                            common=common,
                        )
                        label_curves.extend(curves)
                        label_diagnostics.extend(diagnostics)
                progress.set_postfix({"source": source, "elapsed": f"{time.perf_counter()-started:.1f}s"})
                progress.update(1)
        finally:
            progress.close()
        curves_frame = pd.DataFrame(label_curves)
        diagnostics_frame = pd.DataFrame(label_diagnostics)
        diagnostics_frame = _propagate_rollout_singularity(diagnostics_frame)
        curves_frame.to_csv(out_dir / f"pullback_curves_{label}.csv", index=False)
        diagnostics_frame.to_csv(out_dir / f"pullback_step_diagnostics_{label}.csv", index=False)
        all_curves.append(curves_frame)
        all_diagnostics.append(diagnostics_frame)
    curves = pd.concat(all_curves, ignore_index=True, sort=False)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True, sort=False)
    curves.to_csv(out_dir / "pullback_curves.csv", index=False)
    diagnostics.to_csv(out_dir / "pullback_step_diagnostics.csv", index=False)
    pd.DataFrame(lr_rows).to_csv(out_dir / "pullback_selected_lrs.csv", index=False)
    per_start = _summary(curves, diagnostics, out_dir)
    paired_summary = _paired_contrasts(per_start, out_dir, [float(value) for value in args.trust_fraction])
    gap_closure = _gap_closure_summary(
        per_start,
        out_dir,
        production_mode=bool(args.protocol_mode == "production"),
    )
    _plot_curves(curves, out_dir)
    request = {
        "protocol_version": "pullback_deployment_v3",
        "protocol_mode": str(args.protocol_mode),
        "primary_estimand": (
            f"train step0 loss minus mean train loss over optimizer steps 1..{int(args.steps)}; larger is better"
        ),
        "primary_pullback_fraction": PRIMARY_PULLBACK_FRACTION,
        "primary_gap_recovery_fraction": PRIMARY_GAP_RECOVERY_FRACTION,
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "runs": [str(value) for value in run_dirs],
        "run_checkpoint_sha256": checkpoint_file_hashes,
        "run_config_sha256": [_sha256_file(value / "config.json") for value in run_dirs],
        "run_selected_lrs_sha256": [_sha256_file(value / "selected_lrs.csv") for value in run_dirs],
        "run_weight_pool_sha256": weight_pool_file_hashes,
        "run_weight_pool_records_sha256": weight_record_file_hashes,
        "task_tensor_fingerprints": task_fingerprints,
        "fairness_batch_sizes": fairness_batch_sizes,
        "resolved_lrs": resolved_lrs,
        "labels": labels,
        "start_bank_csv": str(Path(args.start_bank_csv).expanduser().resolve()),
        "start_bank_sha256": _sha256_file(Path(args.start_bank_csv).expanduser().resolve()),
        "selected_start_rows_sha256": _stable_json_hash({"csv": start_bank.to_csv(index=False)}),
        "source_indices": [int(value) for value in start_bank["source_weight_index"].tolist()],
        "methods": list(METHODS),
        "trust_fractions": [float(value) for value in args.trust_fraction],
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "projector_rcond": float(args.projector_rcond),
        "jacobian_chunk_size": int(args.jacobian_chunk_size),
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(args.seed),
    }
    request_hash = _stable_json_hash(request)
    initial = curves[curves["step"] == 0]
    initial_spread = initial.groupby(["label", "source_weight_index"])[["train_loss", "test_loss"]].agg(lambda values: float(values.max() - values.min()))
    trust_methods = [method for method in METHODS if method.endswith("_trust")]
    trust = diagnostics[diagnostics["method"].isin(trust_methods)]
    preimage = diagnostics[diagnostics["method"].isin(["pullback_natural_trust", "projected_raw_adam_trust"])]
    expected_method_fractions = {("raw_adam", 1.0), ("latent_adam_native", 1.0)} | {
        (method, float(fraction))
        for method in METHODS[2:]
        for fraction in args.trust_fraction
    }
    actual_method_fractions = {
        (str(row.method), float(row.trust_fraction))
        for row in curves[["method", "trust_fraction"]].drop_duplicates().itertuples(index=False)
    }
    expected_curve_steps = _expected_eval_steps(int(args.steps), int(args.eval_every))
    expected_sources = [int(value) for value in start_bank["source_weight_index"].tolist()]
    expected_curve_key_set = {
        (label, source, method, float(fraction), step)
        for label in labels
        for source in expected_sources
        for method, fraction in expected_method_fractions
        for step in expected_curve_steps
    }
    actual_curve_key_set = _branch_key_set(curves, step_column="step")
    expected_diagnostic_key_set = {
        (label, source, method, float(fraction), step)
        for label in labels
        for source in expected_sources
        for method, fraction in expected_method_fractions
        for step in range(int(args.steps))
    }
    actual_diagnostic_key_set = _branch_key_set(diagnostics, step_column="proposal_step")
    initial_hash_counts = initial.groupby(["label", "source_weight_index"])["state_theta_sha256"].nunique()
    initial_hash_matches_declared = bool(initial["state_theta_sha256"].eq(initial["theta0_sha256"]).all())
    retraction = diagnostics[
        diagnostics["method"].isin(
            ["latent_sgd_jjt_trust", "pullback_natural_trust", "projected_raw_adam_trust"]
        )
    ]
    applicable_diagnostics_finite = bool(
        np.isfinite(
            diagnostics[["batch_loss", "target_raw_radius", "realized_radius"]].to_numpy(dtype=np.float64)
        ).all()
        and np.isfinite(
            trust[["trust_match_rel_error", "trust_scale", "target_trust_radius"]].to_numpy(dtype=np.float64)
        ).all()
        and np.isfinite(
            preimage[
                [
                    "preimage_algebraic_residual",
                    "preimage_quantization_residual",
                    "preimage_executable_residual",
                    "executed_preimage_algebraic_residual",
                    "executed_preimage_quantization_residual",
                    "executed_preimage_executable_residual",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
        and np.isfinite(
            retraction[["retraction_rel_error", "retraction_cosine"]].to_numpy(dtype=np.float64)
        ).all()
    )
    validation = {
        "request_hash": request_hash,
        "request_payload_rehash_matches": bool(_stable_json_hash(request) == request_hash),
        "adam_max_abs_error": adam_error,
        "all_finite_curves": bool(
            np.isfinite(
                curves[["train_loss", "train_acc", "test_loss", "test_acc", "cumulative_weight_path"]].to_numpy(dtype=np.float64)
            ).all()
        ),
        "all_finite_applicable_diagnostics": applicable_diagnostics_finite,
        "labels": sorted(curves["label"].unique().tolist()),
        "expected_labels": sorted(labels),
        "methods": sorted(curves["method"].unique().tolist()),
        "starts": int(curves["source_weight_index"].nunique()),
        "source_order": expected_sources,
        "steps": int(args.steps),
        "curve_rows": int(len(curves)),
        "expected_curve_rows": int(len(expected_curve_key_set)),
        "diagnostic_rows": int(len(diagnostics)),
        "expected_diagnostic_rows": int(len(expected_diagnostic_key_set)),
        "missing_curve_key_count": int(len(expected_curve_key_set - actual_curve_key_set)),
        "unexpected_curve_key_count": int(len(actual_curve_key_set - expected_curve_key_set)),
        "missing_diagnostic_key_count": int(len(expected_diagnostic_key_set - actual_diagnostic_key_set)),
        "unexpected_diagnostic_key_count": int(len(actual_diagnostic_key_set - expected_diagnostic_key_set)),
        "initial_train_loss_spread_max": float(initial_spread["train_loss"].max()),
        "initial_test_loss_spread_max": float(initial_spread["test_loss"].max()),
        "initial_theta_hash_max_unique": int(initial_hash_counts.max()),
        "initial_theta_hash_matches_declared": initial_hash_matches_declared,
        "curve_start_context_mismatch_count": _context_mismatch_count(curves, start_bank, labels),
        "diagnostic_start_context_mismatch_count": _context_mismatch_count(diagnostics, start_bank, labels),
        "trust_match_rel_error_max": float(trust["trust_match_rel_error"].max()),
        "trust_boundary_hit_count": int(trust["trust_scale_boundary_hit"].astype(bool).sum()),
        "trust_bracket_missing_count": int((~trust["trust_bracket_found"].astype(bool)).sum()),
        "trust_executable_fraction": float(trust["trust_executable"].astype(bool).mean()),
        "method_fraction_grid_matches": bool(actual_method_fractions == expected_method_fractions),
        "preimage_algebraic_residual_max": float(preimage["preimage_algebraic_residual"].max()),
        "preimage_executable_residual_max": float(preimage["preimage_executable_residual"].max()),
        "executed_preimage_algebraic_residual_max": float(preimage["executed_preimage_algebraic_residual"].max()),
        "executed_preimage_executable_residual_max": float(preimage["executed_preimage_executable_residual"].max()),
        "executed_preimage_quantization_consistency": bool(
            (
                preimage["executed_preimage_executable_residual"]
                <= 1.25
                * (
                    preimage["executed_preimage_algebraic_residual"]
                    + preimage["executed_preimage_quantization_residual"]
                )
                + 1.0e-7
            ).all()
        ),
        "operator_executable_fraction": float(preimage["operator_executable"].astype(bool).mean()),
        "retraction_executable_fraction": float(retraction["retraction_executable"].astype(bool).mean()),
        "singularity_step_count": int(preimage["singularity_stratum"].astype(bool).sum()),
        "batch_hash_unique_per_start_step_max": int(diagnostics.groupby(["label", "source_weight_index", "proposal_step"])["batch_indices_sha256"].nunique().max()),
        "duplicate_curve_keys": int(curves.duplicated(["label", "source_weight_index", "method", "trust_fraction", "step"]).sum()),
        "duplicate_diagnostic_keys": int(diagnostics.duplicated(["label", "source_weight_index", "method", "trust_fraction", "proposal_step"]).sum()),
        "per_start_summary_rows": int(len(per_start)),
        "paired_summary_rows": int(len(paired_summary)),
        "gap_closure_rows": int(len(gap_closure)),
        "production_protocol_eligible": bool(
            args.protocol_mode == "production"
            and tuple(labels) == PRODUCTION_LABELS
            and tuple(checkpoint_file_hashes) == PRODUCTION_CHECKPOINT_SHA256
            and tuple(expected_sources) == PRODUCTION_SOURCE_INDICES
        ),
        "weight_pool_hash_count": int(len(set(weight_pool_file_hashes))),
        "weight_record_hash_count": int(len(set(weight_record_file_hashes))),
        "task_tensor_fingerprint_count": int(len(set(task_fingerprints))),
        "fairness_batch_size_count": int(len(set(fairness_batch_sizes))),
    }
    checks = {
        "adam_parity": validation["adam_max_abs_error"] <= 2.0e-7,
        "finite": validation["all_finite_curves"] and validation["all_finite_applicable_diagnostics"],
        "request_hash_integrity": validation["request_payload_rehash_matches"],
        "input_fairness": (
            validation["weight_pool_hash_count"] == 1
            and validation["weight_record_hash_count"] == 1
            and validation["task_tensor_fingerprint_count"] == 1
            and validation["fairness_batch_size_count"] == 1
        ),
        "labels": validation["labels"] == validation["expected_labels"],
        "methods": validation["methods"] == sorted(METHODS),
        "method_fraction_grid": validation["method_fraction_grid_matches"],
        "exact_curve_grid": (
            validation["curve_rows"] == validation["expected_curve_rows"]
            and validation["missing_curve_key_count"] == 0
            and validation["unexpected_curve_key_count"] == 0
        ),
        "exact_diagnostic_grid": (
            validation["diagnostic_rows"] == validation["expected_diagnostic_rows"]
            and validation["missing_diagnostic_key_count"] == 0
            and validation["unexpected_diagnostic_key_count"] == 0
        ),
        "same_initial_state": (
            validation["initial_train_loss_spread_max"] <= 1.0e-7
            and validation["initial_test_loss_spread_max"] <= 1.0e-7
            and validation["initial_theta_hash_max_unique"] == 1
            and validation["initial_theta_hash_matches_declared"]
        ),
        "start_context": (
            validation["curve_start_context_mismatch_count"] == 0
            and validation["diagnostic_start_context_mismatch_count"] == 0
        ),
        "same_batches": validation["batch_hash_unique_per_start_step_max"] == 1,
        "no_duplicates": validation["duplicate_curve_keys"] == 0 and validation["duplicate_diagnostic_keys"] == 0,
        "paired_contrasts_written": validation["paired_summary_rows"] > 0,
        "gap_closure_written": validation["gap_closure_rows"] == len(labels) + 1,
    }
    validation["acceptance_checks"] = checks
    validation["acceptance_pass"] = bool(all(checks.values()))
    (out_dir / "pullback_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "request": request,
        "request_hash": request_hash,
        "elapsed_sec": float(time.perf_counter() - started),
        **request,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(
        f"done acceptance_pass={validation['acceptance_pass']} curves={len(curves)} diagnostics={len(diagnostics)} "
        f"elapsed_sec={time.perf_counter()-started:.2f} output_dir={out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal deployment intervention: native latent Adam versus pullback-natural at raw trust.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-bank-csv", required=True)
    parser.add_argument("--source-index", action="append", type=int, default=[])
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--protocol-mode", choices=("production", "smoke"), default="production")
    parser.add_argument("--steps", type=int, default=PRODUCTION_STEPS)
    parser.add_argument("--eval-every", type=int, default=PRODUCTION_EVAL_EVERY)
    parser.add_argument("--projector-rcond", type=float, default=PRODUCTION_PROJECTOR_RCOND)
    parser.add_argument("--trust-fraction", action="append", type=float, default=None)
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    args.trust_fraction = sorted(set(args.trust_fraction or list(PRODUCTION_TRUST_FRACTIONS)))
    run(args)


if __name__ == "__main__":
    main()
