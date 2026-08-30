#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    decode_weights,
    decoder_jacobians,
    encode_weights,
    logits_from_flat,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress
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


def _log(message: str) -> None:
    print(f"[common_state_rate] {message}", flush=True)


def _norm(value: torch.Tensor) -> float:
    return float(value.detach().double().norm().cpu().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().flatten()
    b = right.detach().double().flatten()
    denom = a.norm() * b.norm()
    if float(denom.detach().cpu().item()) <= 1.0e-30:
        return float("nan")
    return float((torch.dot(a, b) / denom).detach().cpu().item())


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0e-30)


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class AdamState:
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @classmethod
    def zeros_like(cls, value: torch.Tensor) -> AdamState:
        return cls(step=0, exp_avg=torch.zeros_like(value), exp_avg_sq=torch.zeros_like(value))

    def clone(self) -> AdamState:
        return AdamState(self.step, self.exp_avg.clone(), self.exp_avg_sq.clone())


def _adam_delta(
    state: AdamState,
    grad: torch.Tensor,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    state.step += 1
    state.exp_avg.mul_(float(beta1)).add_(grad.detach(), alpha=1.0 - float(beta1))
    state.exp_avg_sq.mul_(float(beta2)).addcmul_(grad.detach(), grad.detach(), value=1.0 - float(beta2))
    correction1 = 1.0 - float(beta1) ** int(state.step)
    correction2 = 1.0 - float(beta2) ** int(state.step)
    denom = state.exp_avg_sq.sqrt().div(math.sqrt(max(correction2, 1.0e-30))).add(float(eps))
    return state.exp_avg.div(denom).mul(-float(lr) / max(correction1, 1.0e-30)).detach()


def _validate_adam(device: torch.device) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(719)
    initial = torch.randn(31, generator=generator, dtype=torch.float32).to(device)
    manual_value = initial.clone()
    torch_value = initial.clone().requires_grad_(True)
    state = AdamState.zeros_like(manual_value)
    optimizer = torch.optim.Adam([torch_value], lr=0.007, betas=(0.9, 0.999), eps=1.0e-8)
    max_error = 0.0
    for _ in range(4):
        grad = torch.randn(31, generator=generator, dtype=torch.float32).to(device)
        manual_value = manual_value + _adam_delta(state, grad, lr=0.007)
        optimizer.zero_grad(set_to_none=True)
        torch_value.grad = grad.clone()
        optimizer.step()
        max_error = max(max_error, float((manual_value - torch_value.detach()).abs().max().cpu().item()))
    return max_error


@dataclass
class OrthogonalProjector:
    jacobian64: torch.Tensor
    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors_t: torch.Tensor
    keep: torch.Tensor
    threshold: float

    def project64(self, vector: torch.Tensor) -> torch.Tensor:
        basis = self.left_vectors[:, self.keep]
        vector64 = vector.detach().double()
        return basis @ (basis.T @ vector64)

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        return self.project64(vector).to(device=vector.device, dtype=vector.dtype)

    def latent_preimage64(self, vector: torch.Tensor) -> torch.Tensor:
        left = self.left_vectors[:, self.keep]
        right = self.right_vectors_t[self.keep, :].T
        target = self.project64(vector)
        coordinate = left.T @ target
        return right @ (coordinate / self.singular_values[self.keep])

    def latent_preimage(self, vector: torch.Tensor) -> torch.Tensor:
        left = self.left_vectors[:, self.keep]
        right = self.right_vectors_t[self.keep, :].T
        target = self.project64(vector)
        latent = self.latent_preimage64(vector).to(
            device=vector.device,
            dtype=vector.dtype,
        )
        best = latent
        best_error = _norm(self.jacobian64 @ best.detach().double() - target)
        for _ in range(3):
            residual = target - self.jacobian64 @ best.detach().double()
            correction = right @ ((left.T @ residual) / self.singular_values[self.keep])
            candidate = (best.detach().double() + correction).to(device=vector.device, dtype=vector.dtype)
            candidate_error = _norm(self.jacobian64 @ candidate.detach().double() - target)
            if candidate_error >= best_error:
                break
            best = candidate
            best_error = candidate_error
        return best

    def preimage_residuals(self, vector: torch.Tensor, latent32: torch.Tensor) -> dict[str, float]:
        target64 = self.project64(vector)
        latent64 = self.latent_preimage64(vector)
        denominator = max(_norm(target64), 1.0e-30)
        algebraic = _norm(self.jacobian64 @ latent64 - target64) / denominator
        quantization = _norm(self.jacobian64 @ (latent32.detach().double() - latent64)) / denominator
        executable = _norm(self.jacobian64 @ latent32.detach().double() - target64) / denominator
        return {
            "algebraic": float(algebraic),
            "quantization": float(quantization),
            "executable": float(executable),
        }

    @property
    def rank(self) -> int:
        return int(self.keep.sum().detach().cpu().item())

    @property
    def mean_nonzero_eigenvalue(self) -> float:
        return float(self.singular_values[self.keep].square().mean().detach().cpu().item())


def _build_projector(jacobian: torch.Tensor, *, rcond: float) -> OrthogonalProjector:
    jacobian64 = jacobian.detach().double()
    left_vectors, singular_values, right_vectors_t = torch.linalg.svd(jacobian64, full_matrices=False)
    largest = float(singular_values.max().detach().cpu().item())
    threshold = max(float(rcond) * max(largest, 0.0), 1.0e-30)
    keep = singular_values > threshold
    if not bool(keep.any().detach().cpu().item()):
        raise RuntimeError(f"rank-revealing projector retained no directions: max_singular={largest:.6g} rcond={rcond:.3g}")
    return OrthogonalProjector(jacobian64, left_vectors, singular_values, right_vectors_t, keep, threshold)


def _projector_with_rcond(projector: OrthogonalProjector, *, rcond: float) -> OrthogonalProjector:
    largest = float(projector.singular_values.max().detach().cpu().item())
    threshold = max(float(rcond) * max(largest, 0.0), 1.0e-30)
    keep = projector.singular_values > threshold
    if not bool(keep.any().detach().cpu().item()):
        raise RuntimeError(f"rank-revealing projector retained no directions: max_singular={largest:.6g} rcond={rcond:.3g}")
    return OrthogonalProjector(
        projector.jacobian64,
        projector.left_vectors,
        projector.singular_values,
        projector.right_vectors_t,
        keep,
        threshold,
    )


def _projector_diagnostics(
    projector: OrthogonalProjector,
    *,
    vector: torch.Tensor,
    jacobian: torch.Tensor,
    rcond: float,
) -> dict[str, float | int]:
    projected = projector.project(vector)
    projected_twice = projector.project(projected)
    normal = vector - projected
    j_norm = max(float(projector.singular_values.max().detach().cpu().item()), 1.0e-30)
    normal_jt = projector.jacobian64.T @ normal.detach().double()
    ranks: dict[str, int] = {}
    largest = float(projector.singular_values.max().detach().cpu().item())
    for multiplier in (0.1, 1.0, 10.0):
        threshold = max(float(rcond) * multiplier * max(largest, 0.0), 1.0e-30)
        ranks[f"projector_rank_rcond_x{multiplier:g}"] = int((projector.singular_values > threshold).sum().cpu().item())
    eigenvalues = projector.singular_values.square()
    return {
        "projector_rank": int(projector.rank),
        "projector_threshold": float(projector.threshold),
        "projector_singular_min_kept": float(projector.singular_values[projector.keep].min().cpu().item()),
        "projector_singular_max": largest,
        "projector_condition_kept": _safe_ratio(largest, float(projector.singular_values[projector.keep].min().cpu().item())),
        "projector_trace": float(eigenvalues.sum().cpu().item()),
        "projector_mean_nonzero_eigenvalue": float(projector.mean_nonzero_eigenvalue),
        "projector_idempotence_rel": _safe_ratio(_norm(projected_twice - projected), _norm(projected)),
        "projector_normal_jt_rel": _safe_ratio(_norm(normal_jt), j_norm * _norm(normal)),
        "projector_tangent_normal_dot_rel": abs(float(torch.dot(projected.double(), normal.double()).cpu().item()))
        / max(_norm(projected) * _norm(normal), 1.0e-30),
        "jacobian_sha256_prefix": int(_sha256_tensor(jacobian)[:12], 16),
        **ranks,
    }


@dataclass
class Candidate:
    name: str
    space: str
    base: torch.Tensor
    linear_delta: torch.Tensor
    native: bool = False
    family: str = ""
    epsilon_fraction: float = float("nan")
    rotation_id: int = -1
    projector_rcond: float = float("nan")
    uses_pseudoinverse: bool = False


@dataclass
class MatchResult:
    scale: float
    metric: float
    theta_new: torch.Tensor
    delta: torch.Tensor
    linear_delta: torch.Tensor
    global_monotone: bool
    boundary_hit: bool
    bracket_found: bool
    local_positive_slope: bool
    crossing_count: int
    bracket_low_scale: float
    bracket_high_scale: float
    bracket_low_metric: float
    bracket_high_metric: float
    iterations: int
    termination_reason: str


TRUST_SCAN_SCALES = (0.0, *(2.0**exponent for exponent in range(-20, 7)))
PROJECTOR_RCONDS = (1.0e-6, 1.0e-5, 1.0e-4)
PRIMARY_PROJECTOR_RCOND = 1.0e-5


def _rcond_tag(value: float) -> str:
    exponent = int(round(-math.log10(float(value))))
    return f"1e-{exponent}"


def _theta_for_scale(
    candidate: Candidate,
    *,
    scale: float,
    theta: torch.Tensor,
    z: torch.Tensor,
    vae: torch.nn.Module,
    normalizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate.space == "ambient":
        delta = candidate.base * float(scale)
        return theta + delta, delta, candidate.linear_delta * float(scale)
    if candidate.space != "latent":
        raise ValueError(f"unknown candidate space {candidate.space!r}")
    with torch.no_grad():
        theta_new = decode_weights(vae, normalizer, (z + candidate.base * float(scale)).reshape(1, -1)).squeeze(0)
    delta = theta_new - theta
    return theta_new, delta, candidate.linear_delta * float(scale)


def _function_rms(
    theta_new: torch.Tensor,
    *,
    base_logits: torch.Tensor,
    calibration_images: torch.Tensor,
    spec: Any,
    tau: float,
) -> float:
    with torch.no_grad():
        logits = logits_from_flat(theta_new, calibration_images, spec, tau=float(tau))
    delta = logits - base_logits
    centered = delta - delta.mean(dim=-1, keepdim=True)
    return float(centered.double().square().mean().sqrt().cpu().item())


def _function_aux_metrics(
    theta_new: torch.Tensor,
    *,
    base_logits: torch.Tensor,
    calibration_images: torch.Tensor,
    spec: Any,
    tau: float,
) -> tuple[float, float, float]:
    with torch.no_grad():
        logits = logits_from_flat(theta_new, calibration_images, spec, tau=float(tau))
        delta = logits - base_logits
        centered_delta = delta - delta.mean(dim=-1, keepdim=True)
        centered = float(centered_delta.double().square().mean().sqrt().cpu().item())
        uncentered = float(delta.double().square().mean().sqrt().cpu().item())
        base_log_prob = F.log_softmax(base_logits, dim=-1)
        new_log_prob = F.log_softmax(logits, dim=-1)
        base_prob = base_log_prob.exp()
        new_prob = new_log_prob.exp()
        symmetric_kl = 0.5 * (
            F.kl_div(new_log_prob, base_prob, reduction="batchmean")
            + F.kl_div(base_log_prob, new_prob, reduction="batchmean")
        )
    return centered, uncentered, float(symmetric_kl.detach().cpu().item())


def _candidate_metric(
    candidate: Candidate,
    *,
    scale: float,
    trust_mode: str,
    theta: torch.Tensor,
    z: torch.Tensor,
    vae: torch.nn.Module,
    normalizer: Any,
    base_logits: torch.Tensor,
    calibration_images: torch.Tensor,
    spec: Any,
    tau: float,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_new, delta, linear_delta = _theta_for_scale(
        candidate,
        scale=float(scale),
        theta=theta,
        z=z,
        vae=vae,
        normalizer=normalizer,
    )
    if trust_mode == "weight":
        metric = _norm(delta)
    elif trust_mode == "function":
        metric = _function_rms(
            theta_new,
            base_logits=base_logits,
            calibration_images=calibration_images,
            spec=spec,
            tau=float(tau),
        )
    else:
        raise ValueError(f"unknown trust_mode={trust_mode!r}")
    return metric, theta_new, delta, linear_delta


def _candidate_trust_scan(
    candidate: Candidate,
    *,
    theta: torch.Tensor,
    z: torch.Tensor,
    vae: torch.nn.Module,
    normalizer: Any,
    base_logits: torch.Tensor,
    calibration_images: torch.Tensor,
    spec: Any,
    tau: float,
) -> dict[str, list[tuple[float, float]]]:
    scans: dict[str, list[tuple[float, float]]] = {"weight": [], "function": []}
    for scale in TRUST_SCAN_SCALES:
        if float(scale) == 0.0:
            weight_metric = 0.0
            function_metric = 0.0
        else:
            theta_new, delta, _linear_delta = _theta_for_scale(
                candidate,
                scale=float(scale),
                theta=theta,
                z=z,
                vae=vae,
                normalizer=normalizer,
            )
            weight_metric = _norm(delta)
            function_metric = _function_rms(
                theta_new,
                base_logits=base_logits,
                calibration_images=calibration_images,
                spec=spec,
                tau=float(tau),
            )
        scans["weight"].append((float(scale), float(weight_metric)))
        scans["function"].append((float(scale), float(function_metric)))
    return scans


def _match_scale(
    candidate: Candidate,
    *,
    target: float,
    trust_mode: str,
    scan: list[tuple[float, float]],
    theta: torch.Tensor,
    z: torch.Tensor,
    vae: torch.nn.Module,
    normalizer: Any,
    base_logits: torch.Tensor,
    calibration_images: torch.Tensor,
    spec: Any,
    tau: float,
) -> MatchResult:
    ordered = sorted((float(scale), float(metric)) for scale, metric in scan)
    finite = all(math.isfinite(scale) and math.isfinite(metric) for scale, metric in ordered)
    global_monotone = finite and all(
        right_metric + max(1.0e-12, 0.01 * abs(left_metric)) >= left_metric
        for (_left_scale, left_metric), (_right_scale, right_metric) in zip(ordered, ordered[1:], strict=False)
    )
    all_crossings = [
        (left, right)
        for left, right in zip(ordered, ordered[1:], strict=False)
        if math.isfinite(left[1])
        and math.isfinite(right[1])
        and (left[1] - float(target)) * (right[1] - float(target)) <= 0.0
        and left[1] != right[1]
    ]
    upward = [(left, right) for left, right in all_crossings if left[1] <= float(target) <= right[1] and right[1] > left[1]]
    bracket_found = bool(upward)
    if bracket_found:
        (low_scale, low_metric), (high_scale, high_metric) = upward[0]
        chosen_scale = low_scale if abs(low_metric - float(target)) <= abs(high_metric - float(target)) else high_scale
        chosen_metric, theta_new, delta, linear_delta = _candidate_metric(
            candidate,
            scale=chosen_scale,
            trust_mode=trust_mode,
            theta=theta,
            z=z,
            vae=vae,
            normalizer=normalizer,
            base_logits=base_logits,
            calibration_images=calibration_images,
            spec=spec,
            tau=float(tau),
        )
        iterations = 0
        termination_reason = "scan_endpoint"
        for iterations in range(1, 41):
            rel_error = abs(float(chosen_metric) - float(target)) / max(float(target), 1.0e-30)
            if rel_error <= 0.0025:
                termination_reason = "matched"
                break
            mid_scale = 0.5 * (float(low_scale) + float(high_scale))
            mid_metric, mid_theta, mid_delta, mid_linear = _candidate_metric(
                candidate,
                scale=mid_scale,
                trust_mode=trust_mode,
                theta=theta,
                z=z,
                vae=vae,
                normalizer=normalizer,
                base_logits=base_logits,
                calibration_images=calibration_images,
                spec=spec,
                tau=float(tau),
            )
            if not math.isfinite(float(mid_metric)):
                termination_reason = "nonfinite_bisection"
                break
            if abs(float(mid_metric) - float(target)) < abs(float(chosen_metric) - float(target)):
                chosen_scale = float(mid_scale)
                chosen_metric = float(mid_metric)
                theta_new, delta, linear_delta = mid_theta, mid_delta, mid_linear
            if float(mid_metric) >= float(target):
                high_scale, high_metric = float(mid_scale), float(mid_metric)
            else:
                low_scale, low_metric = float(mid_scale), float(mid_metric)
        else:
            termination_reason = "max_iterations"
    else:
        low_scale = high_scale = low_metric = high_metric = float("nan")
        chosen_scale, _scan_metric = min(ordered, key=lambda item: abs(item[1] - float(target)))
        chosen_metric, theta_new, delta, linear_delta = _candidate_metric(
            candidate,
            scale=chosen_scale,
            trust_mode=trust_mode,
            theta=theta,
            z=z,
            vae=vae,
            normalizer=normalizer,
            base_logits=base_logits,
            calibration_images=calibration_images,
            spec=spec,
            tau=float(tau),
        )
        iterations = 0
        termination_reason = "no_upward_crossing"
    local_positive_slope = bool(bracket_found and high_metric > low_metric and high_scale > low_scale)
    boundary_hit = bool(chosen_scale >= 63.999)
    return MatchResult(
        scale=float(chosen_scale),
        metric=float(chosen_metric),
        theta_new=theta_new,
        delta=delta,
        linear_delta=linear_delta,
        global_monotone=bool(global_monotone),
        boundary_hit=boundary_hit,
        bracket_found=bracket_found,
        local_positive_slope=local_positive_slope,
        crossing_count=int(len(all_crossings)),
        bracket_low_scale=float(low_scale),
        bracket_high_scale=float(high_scale),
        bracket_low_metric=float(low_metric),
        bracket_high_metric=float(high_metric),
        iterations=int(iterations),
        termination_reason=str(termination_reason),
    )


def _eval_loss(
    theta: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> float:
    with torch.no_grad():
        loss, _ = _loss_acc(
            theta,
            task_set=task_set,
            spec=spec,
            split="train",
            tau=float(tau),
            batch_indices=batch_indices,
        )
    return float(loss.detach().cpu().item())


def _eval_test_loss(theta: torch.Tensor, *, task_set: Any, spec: Any, tau: float) -> float:
    with torch.no_grad():
        loss, _ = _loss_acc(theta, task_set=task_set, spec=spec, split="test", tau=float(tau))
    return float(loss.detach().cpu().item())


def _random_orthogonal(dim: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((int(dim), int(dim)), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(r)).clamp_min(0.0).mul(2.0).sub(1.0)
    q = q * signs.reshape(1, -1)
    return q.to(device=device, dtype=dtype)


def _calibration_indices(
    task_set: Any,
    *,
    batch_size: int,
    stream_start_index: int,
    checkpoint_steps: set[int],
    batch_draws: int,
    count: int,
) -> torch.Tensor:
    train_count = int(task_set.train_labels.shape[0])
    forbidden = torch.zeros(train_count, device=task_set.train_labels.device, dtype=torch.bool)
    for path_step in sorted(checkpoint_steps):
        for draw_id in range(int(batch_draws)):
            for probe_step in (int(path_step) + int(draw_id) * 104729, int(path_step) + (int(draw_id) + 17) * 130363):
                indices = _batch_indices(
                    task_set,
                    batch_size=batch_size,
                    step=probe_step,
                    start_bank_position=stream_start_index,
                )
                if indices is None:
                    raise ValueError("common-state calibration requires minibatched downstream evaluation")
                forbidden[indices] = True
    available = torch.arange(train_count, device=forbidden.device, dtype=torch.long)[~forbidden]
    if int(available.numel()) < int(count):
        raise RuntimeError(
            f"not enough calibration examples disjoint from probe batches: available={int(available.numel())} required={int(count)}"
        )
    return available[-int(count) :].detach()


def _index_overlap(left: torch.Tensor | None, right: torch.Tensor | None) -> int:
    if left is None or right is None:
        return -1
    return int(torch.isin(left.detach(), right.detach()).sum().cpu().item())


def _candidate_rows(
    *,
    candidates: list[Candidate],
    theta: torch.Tensor,
    z: torch.Tensor,
    grad_theta: torch.Tensor,
    vae: torch.nn.Module,
    normalizer: Any,
    task_set: Any,
    spec: Any,
    tau: float,
    update_batch_indices: torch.Tensor | None,
    eval_batch_indices: torch.Tensor | None,
    calibration_images: torch.Tensor,
    radius_fractions: tuple[float, ...],
    reference: Candidate,
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    with torch.no_grad():
        base_logits = logits_from_flat(theta, calibration_images, spec, tau=float(tau)).detach()
    base_update_loss = _eval_loss(theta, task_set=task_set, spec=spec, tau=tau, batch_indices=update_batch_indices)
    base_eval_loss = _eval_loss(theta, task_set=task_set, spec=spec, tau=tau, batch_indices=eval_batch_indices)
    base_full_train_loss = _eval_loss(theta, task_set=task_set, spec=spec, tau=tau, batch_indices=None)
    base_test_loss = _eval_test_loss(theta, task_set=task_set, spec=spec, tau=tau)
    reference_metrics: dict[str, float] = {}
    for trust_mode in ("weight", "function"):
        value, _theta, _delta, _linear = _candidate_metric(
            reference,
            scale=1.0,
            trust_mode=trust_mode,
            theta=theta,
            z=z,
            vae=vae,
            normalizer=normalizer,
            base_logits=base_logits,
            calibration_images=calibration_images,
            spec=spec,
            tau=tau,
        )
        reference_metrics[trust_mode] = value

    rows: list[dict[str, Any]] = []
    schedules: list[tuple[str, float, float]] = []
    for trust_mode in ("weight", "function"):
        for fraction in radius_fractions:
            schedules.append((trust_mode, float(fraction), float(reference_metrics[trust_mode]) * float(fraction)))
    for candidate in candidates:
        trust_scans = _candidate_trust_scan(
            candidate,
            theta=theta,
            z=z,
            vae=vae,
            normalizer=normalizer,
            base_logits=base_logits,
            calibration_images=calibration_images,
            spec=spec,
            tau=tau,
        )
        candidate_schedules = list(schedules)
        if candidate.native:
            candidate_schedules.append(("native", 1.0, float("nan")))
        if candidate is reference:
            candidate_schedules.append(("zero", 0.0, 0.0))
        for trust_mode, radius_fraction, target in candidate_schedules:
            if trust_mode in {"native", "zero"}:
                scale = 1.0 if trust_mode == "native" else 0.0
                theta_new, delta, linear_delta = _theta_for_scale(
                    candidate,
                    scale=scale,
                    theta=theta,
                    z=z,
                    vae=vae,
                    normalizer=normalizer,
                )
                matched_metric = float("nan") if trust_mode == "native" else 0.0
                match_rel_error = float("nan") if trust_mode == "native" else 0.0
                trust_monotone = True
                trust_boundary_hit = False
                trust_bracket_found = True
                trust_local_positive_slope = True
                trust_crossing_count = 1
                trust_bracket_low_scale = float("nan")
                trust_bracket_high_scale = float("nan")
                trust_bracket_low_metric = float("nan")
                trust_bracket_high_metric = float("nan")
                trust_match_iterations = 0
                trust_termination_reason = trust_mode
                trust_scan_json = "[]"
                trust_eligible = True
            else:
                match = _match_scale(
                    candidate,
                    target=target,
                    trust_mode=trust_mode,
                    scan=trust_scans[trust_mode],
                    theta=theta,
                    z=z,
                    vae=vae,
                    normalizer=normalizer,
                    base_logits=base_logits,
                    calibration_images=calibration_images,
                    spec=spec,
                    tau=tau,
                )
                scale = match.scale
                matched_metric = match.metric
                theta_new = match.theta_new
                delta = match.delta
                linear_delta = match.linear_delta
                trust_monotone = match.global_monotone
                trust_boundary_hit = match.boundary_hit
                trust_bracket_found = match.bracket_found
                trust_local_positive_slope = match.local_positive_slope
                trust_crossing_count = match.crossing_count
                trust_bracket_low_scale = match.bracket_low_scale
                trust_bracket_high_scale = match.bracket_high_scale
                trust_bracket_low_metric = match.bracket_low_metric
                trust_bracket_high_metric = match.bracket_high_metric
                trust_match_iterations = match.iterations
                trust_termination_reason = match.termination_reason
                trust_scan_json = json.dumps(trust_scans[trust_mode], separators=(",", ":"))
                match_rel_error = abs(float(matched_metric) - float(target)) / max(float(target), 1.0e-30)
                tolerance = 0.01 if trust_mode == "weight" else 0.02
                trust_eligible = bool(
                    math.isfinite(float(matched_metric))
                    and trust_bracket_found
                    and trust_local_positive_slope
                    and not trust_boundary_hit
                    and float(match_rel_error) <= float(tolerance)
                )
            update_loss = _eval_loss(theta_new, task_set=task_set, spec=spec, tau=tau, batch_indices=update_batch_indices)
            eval_loss = _eval_loss(theta_new, task_set=task_set, spec=spec, tau=tau, batch_indices=eval_batch_indices)
            full_train_loss = _eval_loss(theta_new, task_set=task_set, spec=spec, tau=tau, batch_indices=None)
            test_loss = _eval_test_loss(theta_new, task_set=task_set, spec=spec, tau=tau)
            if candidate.space == "ambient":
                linear_update_loss = update_loss
                linear_eval_loss = eval_loss
                linear_full_train_loss = full_train_loss
                linear_test_loss = test_loss
            else:
                theta_linear = theta + linear_delta
                linear_update_loss = _eval_loss(
                    theta_linear,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    batch_indices=update_batch_indices,
                )
                linear_eval_loss = _eval_loss(
                    theta_linear,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    batch_indices=eval_batch_indices,
                )
                linear_full_train_loss = _eval_loss(
                    theta_linear,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    batch_indices=None,
                )
                linear_test_loss = _eval_test_loss(theta_linear, task_set=task_set, spec=spec, tau=tau)
            function_rms, uncentered_logit_rms, symmetric_kl = _function_aux_metrics(
                theta_new,
                base_logits=base_logits,
                calibration_images=calibration_images,
                spec=spec,
                tau=tau,
            )
            delta_norm = _norm(delta)
            linear_norm = _norm(linear_delta)
            retraction_rel_error = _safe_ratio(_norm(delta - linear_delta), linear_norm)
            retraction_cosine = _cosine(delta, linear_delta)
            retraction_executable = bool(
                candidate.space == "ambient"
                or (
                    math.isfinite(float(retraction_rel_error))
                    and math.isfinite(float(retraction_cosine))
                    and float(retraction_rel_error) <= 0.25
                    and float(retraction_cosine) >= 0.95
                )
            )
            operator_stable = True
            svd_dependent = math.isfinite(float(candidate.projector_rcond))
            if candidate.family == "tangent_ceiling":
                operator_stable = bool(
                    common["tangent_pseudoinverse_stable"]
                    if candidate.uses_pseudoinverse
                    else common["tangent_direction_stable"]
                )
            elif candidate.family == "projected_raw":
                operator_stable = bool(
                    common["projected_pseudoinverse_stable"]
                    if candidate.uses_pseudoinverse
                    else common["projected_direction_stable"]
                )
            elif candidate.family in {"normal_access", "normal_placebo"}:
                operator_stable = bool(common["normal_operator_stable"])
            local_operator_residual_ok = True
            if candidate.uses_pseudoinverse:
                tag = _rcond_tag(float(candidate.projector_rcond)).replace("-", "m")
                residual_name = "tangent" if candidate.family == "tangent_ceiling" else "projected"
                algebraic = float(common[f"rcond_{tag}_{residual_name}_preimage_algebraic_residual"])
                quantization = float(common[f"rcond_{tag}_{residual_name}_preimage_quantization_residual"])
                executable = float(common[f"rcond_{tag}_{residual_name}_preimage_executable_residual"])
                local_operator_residual_ok = bool(
                    algebraic <= 1.0e-10
                    and executable <= 1.0e-4
                    and executable <= 1.25 * (algebraic + quantization) + 1.0e-7
                )
            oracle_execution_eligible = bool(
                trust_eligible and local_operator_residual_ok and retraction_executable
            )
            technical_eligible = bool(
                trust_eligible
                and local_operator_residual_ok
                and (retraction_executable or not candidate.uses_pseudoinverse)
            )
            threshold_specific_eligible = bool(
                technical_eligible and (not svd_dependent or not bool(common["singularity_stratum"]))
            )
            threshold_robust_eligible = bool(
                threshold_specific_eligible and (not svd_dependent or operator_stable)
            )
            g_dot_delta = float(torch.dot(grad_theta.detach().double(), delta.detach().double()).cpu().item())
            update_delta = float(update_loss - base_update_loss)
            rows.append(
                {
                    **common,
                    "candidate": candidate.name,
                    "candidate_family": candidate.family,
                    "candidate_space": candidate.space,
                    "epsilon_fraction": candidate.epsilon_fraction,
                    "rotation_id": candidate.rotation_id,
                    "projector_rcond": candidate.projector_rcond,
                    "uses_pseudoinverse": bool(candidate.uses_pseudoinverse),
                    "trust_mode": trust_mode,
                    "radius_fraction": float(radius_fraction),
                    "target_trust": float(target),
                    "matched_trust": float(matched_metric),
                    "trust_match_rel_error": float(match_rel_error),
                    "trust_response_monotone": bool(trust_monotone),
                    "trust_scale_boundary_hit": bool(trust_boundary_hit),
                    "trust_bracket_found": bool(trust_bracket_found),
                    "trust_local_positive_slope": bool(trust_local_positive_slope),
                    "trust_crossing_count": int(trust_crossing_count),
                    "trust_bracket_low_scale": float(trust_bracket_low_scale),
                    "trust_bracket_high_scale": float(trust_bracket_high_scale),
                    "trust_bracket_low_metric": float(trust_bracket_low_metric),
                    "trust_bracket_high_metric": float(trust_bracket_high_metric),
                    "trust_match_iterations": int(trust_match_iterations),
                    "trust_termination_reason": str(trust_termination_reason),
                    "trust_scan_json": trust_scan_json,
                    "trust_eligible": bool(trust_eligible),
                    "candidate_scale": float(scale),
                    "reference_weight_radius": float(reference_metrics["weight"]),
                    "reference_function_radius": float(reference_metrics["function"]),
                    "delta_weight_norm": delta_norm,
                    "delta_function_rms": function_rms,
                    "delta_logit_rms_uncentered": uncentered_logit_rms,
                    "delta_symmetric_kl": symmetric_kl,
                    "linear_delta_norm": linear_norm,
                    "retraction_rel_error": float(retraction_rel_error),
                    "retraction_cosine": float(retraction_cosine),
                    "retraction_executable": bool(retraction_executable),
                    "operator_threshold_stable": bool(operator_stable),
                    "local_operator_residual_ok": bool(local_operator_residual_ok),
                    "oracle_execution_eligible": bool(oracle_execution_eligible),
                    "technical_eligible": bool(technical_eligible),
                    "threshold_specific_eligible": bool(threshold_specific_eligible),
                    "threshold_robust_eligible": bool(threshold_robust_eligible),
                    "regular_stratum_eligible": bool(threshold_specific_eligible),
                    "cos_with_neg_grad": _cosine(delta, -grad_theta),
                    "g_dot_delta": g_dot_delta,
                    "first_order_progress": -g_dot_delta,
                    "first_order_progress_per_weight_norm": _safe_ratio(-g_dot_delta, delta_norm),
                    "update_batch_loss_base": base_update_loss,
                    "update_batch_loss": update_loss,
                    "update_batch_loss_delta": update_delta,
                    "update_batch_progress": -update_delta,
                    "update_batch_progress_per_weight_norm": _safe_ratio(-update_delta, delta_norm),
                    "second_order_residual": update_delta - g_dot_delta,
                    "directional_curvature_estimate": _safe_ratio(2.0 * (update_delta - g_dot_delta), delta_norm * delta_norm),
                    "eval_batch_loss_base": base_eval_loss,
                    "eval_batch_loss": eval_loss,
                    "eval_batch_loss_delta": eval_loss - base_eval_loss,
                    "full_train_loss_base": base_full_train_loss,
                    "full_train_loss": full_train_loss,
                    "full_train_loss_delta": full_train_loss - base_full_train_loss,
                    "test_loss_base": base_test_loss,
                    "test_loss": test_loss,
                    "test_loss_delta": test_loss - base_test_loss,
                    "same_scale_linear_update_batch_loss_delta": linear_update_loss - base_update_loss,
                    "same_scale_linear_eval_batch_loss_delta": linear_eval_loss - base_eval_loss,
                    "same_scale_linear_full_train_loss_delta": linear_full_train_loss - base_full_train_loss,
                    "same_scale_linear_test_loss_delta": linear_test_loss - base_test_loss,
                    "nonlinear_minus_same_scale_linear_update_batch_loss": update_loss - linear_update_loss,
                    "nonlinear_minus_same_scale_linear_eval_batch_loss": eval_loss - linear_eval_loss,
                    "nonlinear_minus_same_scale_linear_full_train_loss": full_train_loss - linear_full_train_loss,
                    "nonlinear_minus_same_scale_linear_test_loss": test_loss - linear_test_loss,
                }
            )
    return rows


def _probe_state(
    *,
    cfg: Any,
    vae: torch.nn.Module,
    normalizer: Any,
    z: torch.Tensor,
    raw_state: AdamState,
    latent_state: AdamState,
    rotation_states: list[AdamState],
    rotations: list[torch.Tensor],
    raw_lr: float,
    latent_lr: float,
    task_set: Any,
    spec: Any,
    tau: float,
    path_step: int,
    start_bank_position: int,
    stream_start_index: int,
    source_weight_index: int,
    label: str,
    run_name: str,
    draw_id: int,
    calibration_indices: torch.Tensor,
    radius_fractions: tuple[float, ...],
    projector_rcond: float,
    jacobian_chunk_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))
    update_step = int(path_step) + int(draw_id) * 104729
    eval_step = int(path_step) + (int(draw_id) + 17) * 130363
    update_batch = _batch_indices(
        task_set,
        batch_size=batch_size,
        step=update_step,
        start_bank_position=stream_start_index,
    )
    eval_batch = _batch_indices(
        task_set,
        batch_size=batch_size,
        step=eval_step,
        start_bank_position=stream_start_index,
    )
    theta = decode_weights(vae, normalizer, z.detach().reshape(1, -1)).squeeze(0).detach()
    theta_req = theta.clone().requires_grad_(True)
    update_loss, _ = _loss_acc(
        theta_req,
        task_set=task_set,
        spec=spec,
        split="train",
        tau=tau,
        batch_indices=update_batch,
    )
    grad_theta = torch.autograd.grad(update_loss, theta_req)[0].detach()
    jacobian = decoder_jacobians(
        vae,
        normalizer,
        None,
        z.detach().reshape(1, -1),
        create_graph=False,
        chunk_size=int(jacobian_chunk_size),
    ).squeeze(0).detach()
    decomposition = _build_projector(jacobian, rcond=min(PROJECTOR_RCONDS))
    projectors = {rcond: _projector_with_rcond(decomposition, rcond=rcond) for rcond in PROJECTOR_RCONDS}
    projector = projectors[float(projector_rcond)]
    neg_grad = -grad_theta
    tangent_neg_grad = projector.project(neg_grad)
    normal_neg_grad = neg_grad - tangent_neg_grad
    grad_z = jacobian.T @ grad_theta
    jjt_neg_grad = jacobian @ (jacobian.T @ neg_grad)

    raw_replay_state = raw_state.clone()
    raw_replay_delta = _adam_delta(raw_replay_state, grad_theta, lr=float(raw_lr))
    raw_fresh_delta = _adam_delta(AdamState.zeros_like(grad_theta), grad_theta, lr=float(raw_lr))
    latent_replay_state = latent_state.clone()
    latent_replay_dz = _adam_delta(latent_replay_state, grad_z, lr=float(latent_lr))
    latent_fresh_dz = _adam_delta(AdamState.zeros_like(grad_z), grad_z, lr=float(latent_lr))
    tangent_oracle_dz = projector.latent_preimage(neg_grad)
    projected_raw_dz = projector.latent_preimage(raw_replay_delta)

    mean_eigenvalue = float(decomposition.singular_values.square().sum().cpu().item()) / float(jacobian.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1009 * source_weight_index + 9176 * path_step + 7919 * draw_id)
    random_vector = torch.randn(int(grad_theta.numel()), generator=generator, dtype=torch.float32).to(
        device=grad_theta.device,
        dtype=grad_theta.dtype,
    )
    random_normal = random_vector - projector.project(random_vector)
    normal_norm2 = float(torch.dot(normal_neg_grad.double(), normal_neg_grad.double()).cpu().item())
    if normal_norm2 > 1.0e-30:
        random_normal = random_normal - normal_neg_grad * (
            torch.dot(random_normal.double(), normal_neg_grad.double()).to(random_normal.dtype) / normal_norm2
        )
    random_normal = random_normal * (_norm(normal_neg_grad) / max(_norm(random_normal), 1.0e-30))

    candidates: list[Candidate] = [
        Candidate("raw_adam_replay", "ambient", raw_replay_delta, raw_replay_delta, True, "raw_adam"),
        Candidate("raw_adam_fresh", "ambient", raw_fresh_delta, raw_fresh_delta, True, "raw_adam"),
        Candidate("jjt_direct_latent_sgd_linear", "ambient", jjt_neg_grad, jjt_neg_grad, False, "jjt"),
        Candidate("latent_sgd_nonlinear", "latent", jacobian.T @ neg_grad, jjt_neg_grad, False, "jjt"),
        Candidate("latent_adam_replay_linear", "ambient", jacobian @ latent_replay_dz, jacobian @ latent_replay_dz, True, "latent_adam"),
        Candidate("latent_adam_replay_nonlinear", "latent", latent_replay_dz, jacobian @ latent_replay_dz, True, "latent_adam"),
        Candidate("latent_adam_fresh_linear", "ambient", jacobian @ latent_fresh_dz, jacobian @ latent_fresh_dz, True, "latent_adam"),
        Candidate("latent_adam_fresh_nonlinear", "latent", latent_fresh_dz, jacobian @ latent_fresh_dz, True, "latent_adam"),
    ]
    rcond_vectors: dict[float, dict[str, torch.Tensor]] = {}
    for rcond, current_projector in projectors.items():
        tag = _rcond_tag(rcond)
        tangent = current_projector.project(neg_grad)
        normal = neg_grad - tangent
        projected_raw = current_projector.project(raw_replay_delta)
        tangent_preimage = current_projector.latent_preimage(neg_grad)
        projected_raw_preimage = current_projector.latent_preimage(raw_replay_delta)
        current_random_normal = random_vector - current_projector.project(random_vector)
        current_normal_norm2 = float(torch.dot(normal.double(), normal.double()).cpu().item())
        if current_normal_norm2 > 1.0e-30:
            current_random_normal = current_random_normal - normal * (
                torch.dot(current_random_normal.double(), normal.double()).to(current_random_normal.dtype)
                / current_normal_norm2
            )
        current_random_normal = current_random_normal * (
            _norm(normal) / max(_norm(current_random_normal), 1.0e-30)
        )
        rcond_vectors[rcond] = {
            "tangent": tangent,
            "normal": normal,
            "projected_raw": projected_raw,
            "tangent_preimage": tangent_preimage,
            "projected_raw_preimage": projected_raw_preimage,
            "random_normal": current_random_normal,
        }
        candidates.extend(
            [
                Candidate(
                    f"tangent_oracle_rcond{tag}",
                    "ambient",
                    tangent,
                    tangent,
                    False,
                    "tangent_ceiling",
                    projector_rcond=float(rcond),
                ),
                Candidate(
                    f"tangent_oracle_rcond{tag}_nonlinear",
                    "latent",
                    tangent_preimage,
                    tangent,
                    False,
                    "tangent_ceiling",
                    projector_rcond=float(rcond),
                    uses_pseudoinverse=True,
                ),
                Candidate(
                    f"projected_raw_adam_rcond{tag}",
                    "ambient",
                    projected_raw,
                    projected_raw,
                    False,
                    "projected_raw",
                    projector_rcond=float(rcond),
                ),
                Candidate(
                    f"projected_raw_adam_rcond{tag}_nonlinear",
                    "latent",
                    projected_raw_preimage,
                    projected_raw,
                    False,
                    "projected_raw",
                    projector_rcond=float(rcond),
                    uses_pseudoinverse=True,
                ),
            ]
        )
        for epsilon_fraction in (0.25, 1.0):
            epsilon = mean_eigenvalue * float(epsilon_fraction)
            direction = jjt_neg_grad + epsilon * normal
            random_direction = jjt_neg_grad + epsilon * current_random_normal
            candidates.extend(
                [
                    Candidate(
                        f"jjt_plus_normal_rcond{tag}_eps{epsilon_fraction:g}",
                        "ambient",
                        direction,
                        direction,
                        False,
                        "normal_access",
                        float(epsilon_fraction),
                        projector_rcond=float(rcond),
                    ),
                    Candidate(
                        f"jjt_plus_random_normal_placebo_rcond{tag}_eps{epsilon_fraction:g}",
                        "ambient",
                        random_direction,
                        random_direction,
                        False,
                        "normal_placebo",
                        float(epsilon_fraction),
                        projector_rcond=float(rcond),
                    ),
                ]
            )
    for rotation_id, (rotation, rotation_state) in enumerate(zip(rotations, rotation_states, strict=True)):
        rotation_name = "identity" if rotation_id == 0 else f"dense{rotation_id - 1}"
        grad_y = rotation.T @ grad_z
        replay_dy = _adam_delta(rotation_state.clone(), grad_y, lr=float(latent_lr))
        dz = rotation @ replay_dy
        candidates.extend(
            [
                Candidate(
                    f"latent_adam_rotation_{rotation_name}_linear",
                    "ambient",
                    jacobian @ dz,
                    jacobian @ dz,
                    True,
                    "latent_rotation",
                    rotation_id=rotation_id,
                ),
                Candidate(
                    f"latent_adam_rotation_{rotation_name}_nonlinear",
                    "latent",
                    dz,
                    jacobian @ dz,
                    True,
                    "latent_rotation",
                    rotation_id=rotation_id,
                ),
            ]
        )

    calibration_images = task_set.train_images.index_select(0, calibration_indices)
    projector_diag = _projector_diagnostics(
        projector,
        vector=neg_grad,
        jacobian=jacobian,
        rcond=float(projector_rcond),
    )
    primary_vectors = rcond_vectors[float(projector_rcond)]
    sensitivity: dict[str, Any] = {}
    tangent_direction_stable = True
    tangent_preimage_stable = True
    projected_direction_stable = True
    projected_preimage_stable = True
    normal_stable = True
    for rcond, current_projector in projectors.items():
        tag = _rcond_tag(rcond).replace("-", "m")
        vectors = rcond_vectors[rcond]
        retained_min = float(current_projector.singular_values[current_projector.keep].min().cpu().item())
        retained_max = float(current_projector.singular_values.max().cpu().item())
        tangent_residuals = current_projector.preimage_residuals(neg_grad, vectors["tangent_preimage"])
        projected_residuals = current_projector.preimage_residuals(
            raw_replay_delta,
            vectors["projected_raw_preimage"],
        )
        tangent_norm_ratio = _safe_ratio(_norm(vectors["tangent"]), _norm(primary_vectors["tangent"]))
        tangent_preimage_norm_ratio = _safe_ratio(
            _norm(vectors["tangent_preimage"]), _norm(primary_vectors["tangent_preimage"])
        )
        projected_norm_ratio = _safe_ratio(
            _norm(vectors["projected_raw"]), _norm(primary_vectors["projected_raw"])
        )
        projected_preimage_norm_ratio = _safe_ratio(
            _norm(vectors["projected_raw_preimage"]), _norm(primary_vectors["projected_raw_preimage"])
        )
        normal_norm_ratio = _safe_ratio(_norm(vectors["normal"]), _norm(primary_vectors["normal"]))
        placebo_norm_ratio = _safe_ratio(
            _norm(vectors["random_normal"]), _norm(primary_vectors["random_normal"])
        )
        tangent_cosine = _cosine(vectors["tangent"], primary_vectors["tangent"])
        tangent_preimage_cosine = _cosine(
            vectors["tangent_preimage"], primary_vectors["tangent_preimage"]
        )
        projected_cosine = _cosine(vectors["projected_raw"], primary_vectors["projected_raw"])
        projected_preimage_cosine = _cosine(
            vectors["projected_raw_preimage"], primary_vectors["projected_raw_preimage"]
        )
        normal_cosine = _cosine(vectors["normal"], primary_vectors["normal"])
        placebo_cosine = _cosine(vectors["random_normal"], primary_vectors["random_normal"])
        tangent_direction_stable = tangent_direction_stable and bool(
            tangent_cosine >= 0.99
            and 0.9 <= tangent_norm_ratio <= 1.1
        )
        tangent_preimage_stable = tangent_preimage_stable and bool(
            tangent_cosine >= 0.99
            and 0.9 <= tangent_norm_ratio <= 1.1
            and tangent_preimage_cosine >= 0.99
            and 0.8 <= tangent_preimage_norm_ratio <= 1.25
            and tangent_residuals["algebraic"] <= 1.0e-10
            and tangent_residuals["executable"] <= 1.0e-4
            and tangent_residuals["executable"]
            <= 1.25 * (tangent_residuals["algebraic"] + tangent_residuals["quantization"]) + 1.0e-7
        )
        projected_direction_stable = projected_direction_stable and bool(
            projected_cosine >= 0.99
            and 0.9 <= projected_norm_ratio <= 1.1
        )
        projected_preimage_stable = projected_preimage_stable and bool(
            projected_cosine >= 0.99
            and 0.9 <= projected_norm_ratio <= 1.1
            and projected_preimage_cosine >= 0.99
            and 0.8 <= projected_preimage_norm_ratio <= 1.25
            and projected_residuals["algebraic"] <= 1.0e-10
            and projected_residuals["executable"] <= 1.0e-4
            and projected_residuals["executable"]
            <= 1.25 * (projected_residuals["algebraic"] + projected_residuals["quantization"]) + 1.0e-7
        )
        normal_stable = normal_stable and bool(
            normal_cosine >= 0.99
            and 0.9 <= normal_norm_ratio <= 1.1
            and placebo_cosine >= 0.99
            and 0.9 <= placebo_norm_ratio <= 1.1
        )
        sensitivity.update(
            {
                f"rcond_{tag}_rank": int(current_projector.rank),
                f"rcond_{tag}_condition": _safe_ratio(retained_max, retained_min),
                f"rcond_{tag}_tangent_cos_primary": float(tangent_cosine),
                f"rcond_{tag}_tangent_norm_ratio_primary": float(tangent_norm_ratio),
                f"rcond_{tag}_tangent_preimage_cos_primary": float(tangent_preimage_cosine),
                f"rcond_{tag}_tangent_preimage_norm_ratio_primary": float(tangent_preimage_norm_ratio),
                f"rcond_{tag}_tangent_preimage_algebraic_residual": float(tangent_residuals["algebraic"]),
                f"rcond_{tag}_tangent_preimage_quantization_residual": float(tangent_residuals["quantization"]),
                f"rcond_{tag}_tangent_preimage_executable_residual": float(tangent_residuals["executable"]),
                f"rcond_{tag}_tangent_preimage_residual": float(tangent_residuals["executable"]),
                f"rcond_{tag}_projected_cos_primary": float(projected_cosine),
                f"rcond_{tag}_projected_norm_ratio_primary": float(projected_norm_ratio),
                f"rcond_{tag}_projected_preimage_cos_primary": float(projected_preimage_cosine),
                f"rcond_{tag}_projected_preimage_norm_ratio_primary": float(projected_preimage_norm_ratio),
                f"rcond_{tag}_projected_preimage_algebraic_residual": float(projected_residuals["algebraic"]),
                f"rcond_{tag}_projected_preimage_quantization_residual": float(projected_residuals["quantization"]),
                f"rcond_{tag}_projected_preimage_executable_residual": float(projected_residuals["executable"]),
                f"rcond_{tag}_projected_preimage_residual": float(projected_residuals["executable"]),
                f"rcond_{tag}_normal_cos_primary": float(normal_cosine),
                f"rcond_{tag}_normal_norm_ratio_primary": float(normal_norm_ratio),
                f"rcond_{tag}_placebo_cos_primary": float(placebo_cosine),
                f"rcond_{tag}_placebo_norm_ratio_primary": float(placebo_norm_ratio),
            }
        )
    singularity_stratum = bool(
        float(sensitivity["rcond_1em6_condition"]) > 1.0e5
        or int(sensitivity["rcond_1em6_rank"]) != int(sensitivity["rcond_1em5_rank"])
    )
    common = {
        "label": label,
        "run_name": run_name,
        "source_weight_index": int(source_weight_index),
        "start_bank_position": int(start_bank_position),
        "stream_start_index": int(stream_start_index),
        "path_step": int(path_step),
        "draw_id": int(draw_id),
        "task_name": str(task_set.task_name),
        "tau": float(tau),
        "raw_lr": float(raw_lr),
        "latent_lr": float(latent_lr),
        "update_batch_sha256": _tensor_sha256(update_batch),
        "eval_batch_sha256": _tensor_sha256(eval_batch),
        "calibration_images_sha256": _sha256_tensor(calibration_images),
        "calibration_indices_sha256": _tensor_sha256(calibration_indices),
        "update_eval_index_overlap": _index_overlap(update_batch, eval_batch),
        "update_calibration_index_overlap": _index_overlap(update_batch, calibration_indices),
        "eval_calibration_index_overlap": _index_overlap(eval_batch, calibration_indices),
        "theta_sha256": _sha256_tensor(theta),
        "z_sha256": _sha256_tensor(z),
        "path_adam_step": int(latent_state.step),
        "update_batch_loss_at_state": float(update_loss.detach().cpu().item()),
        "grad_theta_norm": _norm(grad_theta),
        "grad_z_norm": _norm(grad_z),
        "tangent_neg_grad_norm": _norm(tangent_neg_grad),
        "normal_neg_grad_norm": _norm(normal_neg_grad),
        "tangent_first_order_ceiling_fraction": _safe_ratio(_norm(tangent_neg_grad), _norm(neg_grad)),
        "normal_residual_fraction": _safe_ratio(_norm(normal_neg_grad), _norm(neg_grad)),
        "raw_adam_tangent_norm_fraction": _safe_ratio(_norm(projector.project(raw_replay_delta)), _norm(raw_replay_delta)),
        "raw_adam_tangent_first_order_fraction": _safe_ratio(
            abs(float(torch.dot(grad_theta.double(), projector.project(raw_replay_delta).double()).cpu().item())),
            abs(float(torch.dot(grad_theta.double(), raw_replay_delta.double()).cpu().item())),
        ),
        "jjt_cos_with_tangent_oracle": _cosine(jjt_neg_grad, tangent_neg_grad),
        "random_normal_task_normal_cos": _cosine(random_normal, normal_neg_grad),
        "tangent_oracle_preimage_norm": _norm(tangent_oracle_dz),
        "projected_raw_preimage_norm": _norm(projected_raw_dz),
        "singularity_stratum": bool(singularity_stratum),
        "tangent_direction_stable": bool(tangent_direction_stable),
        "tangent_pseudoinverse_stable": bool(tangent_preimage_stable),
        "projected_direction_stable": bool(projected_direction_stable),
        "projected_pseudoinverse_stable": bool(projected_preimage_stable),
        "normal_operator_stable": bool(normal_stable),
        "threshold_independent_mean_jtj_eigenvalue": float(mean_eigenvalue),
        **sensitivity,
        **projector_diag,
    }
    rows = _candidate_rows(
        candidates=candidates,
        theta=theta,
        z=z,
        grad_theta=grad_theta,
        vae=vae,
        normalizer=normalizer,
        task_set=task_set,
        spec=spec,
        tau=tau,
        update_batch_indices=update_batch,
        eval_batch_indices=eval_batch,
        calibration_images=calibration_images,
        radius_fractions=radius_fractions,
        reference=candidates[0],
        common=common,
    )
    return rows, common


def _run_start(
    *,
    cfg: Any,
    vae: torch.nn.Module,
    normalizer: Any,
    weights_device: torch.Tensor,
    start_row: pd.Series,
    task_tensors: dict[str, Any],
    spec: Any,
    label: str,
    run_name: str,
    raw_lr: float,
    latent_lr: float,
    trajectory_steps: int,
    checkpoint_steps: set[int],
    batch_draws: int,
    radius_fractions: tuple[float, ...],
    projector_rcond: float,
    jacobian_chunk_size: int,
    rotations: list[torch.Tensor],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_weight_index = int(start_row["source_weight_index"])
    start_bank_position = int(start_row["start_bank_position"])
    stream_start_index = int(start_row.get("start_index", start_bank_position))
    task_name = str(start_row["task_name"])
    tau = float(start_row["tau"])
    task_set = _task_tensor_set(task_tensors, task_name)
    w0 = weights_device[source_weight_index].detach()
    with torch.no_grad():
        z = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
    raw_state = AdamState.zeros_like(w0)
    latent_state = AdamState.zeros_like(z)
    rotation_states = [AdamState.zeros_like(z) for _ in rotations]
    rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))
    calibration_indices = _calibration_indices(
        task_set,
        batch_size=batch_size,
        stream_start_index=stream_start_index,
        checkpoint_steps=checkpoint_steps,
        batch_draws=batch_draws,
        count=min(512, int(task_set.train_images.shape[0]) // 4),
    )
    for path_step in range(int(trajectory_steps) + 1):
        if int(path_step) in checkpoint_steps:
            for draw_id in range(int(batch_draws)):
                candidate_rows, state_row = _probe_state(
                    cfg=cfg,
                    vae=vae,
                    normalizer=normalizer,
                    z=z,
                    raw_state=raw_state,
                    latent_state=latent_state,
                    rotation_states=rotation_states,
                    rotations=rotations,
                    raw_lr=raw_lr,
                    latent_lr=latent_lr,
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    path_step=path_step,
                    start_bank_position=start_bank_position,
                    stream_start_index=stream_start_index,
                    source_weight_index=source_weight_index,
                    label=label,
                    run_name=run_name,
                    draw_id=draw_id,
                    calibration_indices=calibration_indices,
                    radius_fractions=radius_fractions,
                    projector_rcond=projector_rcond,
                    jacobian_chunk_size=jacobian_chunk_size,
                    seed=seed,
                )
                rows.extend(candidate_rows)
                state_rows.append(state_row)
        if int(path_step) == int(trajectory_steps):
            break
        batch_indices = _batch_indices(
            task_set,
            batch_size=batch_size,
            step=int(path_step),
            start_bank_position=stream_start_index,
        )
        z_req = z.detach().clone().requires_grad_(True)
        theta_req = decode_weights(vae, normalizer, z_req.reshape(1, -1)).squeeze(0)
        loss, _ = _loss_acc(
            theta_req,
            task_set=task_set,
            spec=spec,
            split="train",
            tau=tau,
            batch_indices=batch_indices,
        )
        grad_z = torch.autograd.grad(loss, z_req)[0].detach()
        theta_leaf = theta_req.detach().clone().requires_grad_(True)
        raw_loss, _ = _loss_acc(
            theta_leaf,
            task_set=task_set,
            spec=spec,
            split="train",
            tau=tau,
            batch_indices=batch_indices,
        )
        grad_theta = torch.autograd.grad(raw_loss, theta_leaf)[0].detach()
        _adam_delta(raw_state, grad_theta, lr=raw_lr)
        for rotation, rotation_state in zip(rotations, rotation_states, strict=True):
            _adam_delta(rotation_state, rotation.T @ grad_z, lr=latent_lr)
        z = (z + _adam_delta(latent_state, grad_z, lr=latent_lr)).detach()
    return rows, state_rows


def _propagate_start_singularity(
    rows: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    states = states.copy()
    keys = ["label", "source_weight_index"]
    states["singularity_state_local"] = states["singularity_stratum"].astype(bool)
    start_flags = (
        states.groupby(keys, as_index=False)["singularity_state_local"]
        .any()
        .rename(columns={"singularity_state_local": "singularity_start_any_state"})
    )
    states = states.drop(columns=["singularity_stratum"]).merge(start_flags, on=keys, validate="many_to_one")
    states["singularity_stratum"] = states["singularity_start_any_state"].astype(bool)
    rows["singularity_state_local"] = rows["singularity_stratum"].astype(bool)
    rows = rows.drop(columns=["singularity_stratum"]).merge(start_flags, on=keys, validate="many_to_one")
    rows["singularity_stratum"] = rows["singularity_start_any_state"].astype(bool)
    svd_dependent = pd.to_numeric(rows["projector_rcond"], errors="coerce").notna()
    rows["threshold_specific_eligible"] = rows["technical_eligible"].astype(bool) & (
        ~svd_dependent | ~rows["singularity_stratum"].astype(bool)
    )
    rows["threshold_robust_eligible"] = rows["threshold_specific_eligible"].astype(bool) & (
        ~svd_dependent | rows["operator_threshold_stable"].astype(bool)
    )
    rows["regular_stratum_eligible"] = rows["threshold_specific_eligible"].astype(bool)
    return rows, states


def _bootstrap_ci(values: np.ndarray, *, seed: int, reps: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, values.size, size=(int(reps), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summaries(
    rows: pd.DataFrame,
    out_dir: Path,
    *,
    eligibility_column: str = "threshold_robust_eligible",
    suffix: str = "",
    write_coverage: bool = True,
) -> pd.DataFrame:
    trust_rows = rows[rows["trust_mode"].isin(["weight", "function", "native"])].copy()
    coverage = (
        trust_rows.groupby(
            ["label", "candidate", "path_step", "trust_mode", "radius_fraction"],
            dropna=False,
        )
        .agg(
            rows=("source_weight_index", "size"),
            starts=("source_weight_index", "nunique"),
            trust_eligible_rows=("trust_eligible", "sum"),
            technical_eligible_rows=("technical_eligible", "sum"),
            oracle_execution_eligible_rows=("oracle_execution_eligible", "sum"),
            threshold_specific_eligible_rows=("threshold_specific_eligible", "sum"),
            threshold_robust_eligible_rows=("threshold_robust_eligible", "sum"),
            regular_eligible_rows=("regular_stratum_eligible", "sum"),
            singular_rows=("singularity_stratum", "sum"),
        )
        .reset_index()
    )
    if write_coverage:
        coverage.to_csv(out_dir / "common_state_eligibility_coverage.csv", index=False)
    analysis = trust_rows[trust_rows[eligibility_column].astype(bool)].copy()
    group_cols = ["label", "candidate", "candidate_family", "path_step", "trust_mode", "radius_fraction"]
    summary_rows: list[dict[str, Any]] = []
    metrics = [
        "first_order_progress_per_weight_norm",
        "update_batch_progress_per_weight_norm",
        "eval_batch_loss_delta",
        "full_train_loss_delta",
        "test_loss_delta",
        "retraction_rel_error",
        "nonlinear_minus_same_scale_linear_update_batch_loss",
        "nonlinear_minus_same_scale_linear_eval_batch_loss",
        "nonlinear_minus_same_scale_linear_full_train_loss",
        "nonlinear_minus_same_scale_linear_test_loss",
        "trust_match_rel_error",
    ]
    for keys, group in analysis.groupby(group_cols, dropna=False):
        common = dict(zip(group_cols, keys, strict=True))
        for metric in metrics:
            clustered = (
                group.assign(_metric=pd.to_numeric(group[metric], errors="coerce"))
                .groupby("source_weight_index", as_index=False)["_metric"]
                .mean()
                .dropna()
            )
            values = clustered["_metric"].to_numpy(dtype=np.float64)
            lo, hi = _bootstrap_ci(values, seed=1709 + len(summary_rows))
            summary_rows.append(
                {
                    **common,
                    "metric": metric,
                    "n_starts": int(values.size),
                    "n_rows_before_start_aggregation": int(len(group)),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "positive_count": int(np.sum(values > 0.0)),
                    "negative_count": int(np.sum(values < 0.0)),
                    "eligibility_scope": eligibility_column,
                    "inference_unit": "downstream_start_within_vae_checkpoint",
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"common_state_rate_summary{suffix}.csv", index=False)

    pivot_keys = ["label", "source_weight_index", "start_bank_position", "path_step", "draw_id", "trust_mode", "radius_fraction"]
    pairs: list[dict[str, Any]] = []
    comparisons = {
        "latent_replay_minus_raw": ("latent_adam_replay_nonlinear", "raw_adam_replay"),
        "tangent_oracle_minus_raw": ("tangent_oracle_rcond1e-5", "raw_adam_replay"),
        "tangent_oracle_nonlinear_minus_raw": ("tangent_oracle_rcond1e-5_nonlinear", "raw_adam_replay"),
        "projected_raw_nonlinear_minus_raw": ("projected_raw_adam_rcond1e-5_nonlinear", "raw_adam_replay"),
        "jjt_minus_tangent_oracle": ("jjt_direct_latent_sgd_linear", "tangent_oracle_rcond1e-5"),
        "raw_replay_minus_fresh": ("raw_adam_replay", "raw_adam_fresh"),
        "latent_replay_minus_fresh": ("latent_adam_replay_nonlinear", "latent_adam_fresh_nonlinear"),
        "rotation_identity_minus_latent": ("latent_adam_rotation_identity_nonlinear", "latent_adam_replay_nonlinear"),
        "normal_eps1_minus_jjt": ("jjt_plus_normal_rcond1e-5_eps1", "jjt_direct_latent_sgd_linear"),
        "normal_eps0.25_minus_jjt": ("jjt_plus_normal_rcond1e-5_eps0.25", "jjt_direct_latent_sgd_linear"),
        "task_normal_minus_placebo_eps1": (
            "jjt_plus_normal_rcond1e-5_eps1",
            "jjt_plus_random_normal_placebo_rcond1e-5_eps1",
        ),
        "task_normal_minus_placebo_eps0.25": (
            "jjt_plus_normal_rcond1e-5_eps0.25",
            "jjt_plus_random_normal_placebo_rcond1e-5_eps0.25",
        ),
    }
    for rcond in PROJECTOR_RCONDS:
        tag = _rcond_tag(rcond)
        comparisons[f"tangent_rcond{tag}_minus_raw"] = (
            f"tangent_oracle_rcond{tag}",
            "raw_adam_replay",
        )
        comparisons[f"tangent_nonlinear_rcond{tag}_minus_raw"] = (
            f"tangent_oracle_rcond{tag}_nonlinear",
            "raw_adam_replay",
        )
        comparisons[f"projected_raw_nonlinear_rcond{tag}_minus_raw"] = (
            f"projected_raw_adam_rcond{tag}_nonlinear",
            "raw_adam_replay",
        )
    for dense_id in range(max(0, int(rows["rotation_id"].max())) + 1):
        dense_name = f"latent_adam_rotation_dense{dense_id}_nonlinear"
        if bool((rows["candidate"] == dense_name).any()):
            comparisons[f"rotation_dense{dense_id}_minus_identity"] = (dense_name, "latent_adam_rotation_identity_nonlinear")

    contrast_metrics = ["update_batch_loss_delta", "eval_batch_loss_delta", "full_train_loss_delta", "test_loss_delta"]
    clustered_contrast_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for metric in contrast_metrics:
        pivot = analysis.pivot_table(index=pivot_keys, columns="candidate", values=metric, aggfunc="first")
        for contrast, (left, right) in comparisons.items():
            if left not in pivot.columns or right not in pivot.columns:
                continue
            values = (pivot[left] - pivot[right]).rename("contrast_value").dropna().reset_index()
            clustered = (
                values.groupby(
                    ["label", "source_weight_index", "path_step", "trust_mode", "radius_fraction"],
                    as_index=False,
                )["contrast_value"]
                .mean()
            )
            clustered_contrast_cache[(metric, contrast)] = clustered
            for (label, path_step, trust_mode, radius_fraction), group in clustered.groupby(
                ["label", "path_step", "trust_mode", "radius_fraction"]
            ):
                array = group["contrast_value"].to_numpy(dtype=np.float64)
                lo, hi = _bootstrap_ci(array, seed=2719 + len(pairs))
                pairs.append(
                    {
                        "contrast": contrast,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "label": label,
                        "path_step": int(path_step),
                        "trust_mode": trust_mode,
                        "radius_fraction": float(radius_fraction),
                        "n_starts": int(array.size),
                        "mean": float(np.mean(array)),
                        "median": float(np.median(array)),
                        "bootstrap95_low": lo,
                        "bootstrap95_high": hi,
                        "left_worse_count": int(np.sum(array > 0.0)),
                        "left_better_count": int(np.sum(array < 0.0)),
                        "eligibility_scope": eligibility_column,
                        "inference_unit": "downstream_start_within_vae_checkpoint",
                    }
                )

    for metric in contrast_metrics:
        latent = clustered_contrast_cache.get((metric, "latent_replay_minus_fresh"))
        raw = clustered_contrast_cache.get((metric, "raw_replay_minus_fresh"))
        if latent is None or raw is None:
            continue
        keys = ["label", "source_weight_index", "path_step", "trust_mode", "radius_fraction"]
        joined = latent.merge(raw, on=keys, suffixes=("_latent", "_raw"))
        joined["contrast_value"] = joined["contrast_value_latent"] - joined["contrast_value_raw"]
        for (label, path_step, trust_mode, radius_fraction), group in joined.groupby(
            ["label", "path_step", "trust_mode", "radius_fraction"]
        ):
            array = group["contrast_value"].to_numpy(dtype=np.float64)
            lo, hi = _bootstrap_ci(array, seed=3719 + len(pairs))
            pairs.append(
                {
                    "contrast": "history_difference_in_differences",
                    "left": "latent(replay-fresh)",
                    "right": "raw(replay-fresh)",
                    "metric": metric,
                    "label": label,
                    "path_step": int(path_step),
                    "trust_mode": trust_mode,
                    "radius_fraction": float(radius_fraction),
                    "n_starts": int(array.size),
                    "mean": float(np.mean(array)),
                    "median": float(np.median(array)),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "left_worse_count": int(np.sum(array > 0.0)),
                    "left_better_count": int(np.sum(array < 0.0)),
                    "eligibility_scope": eligibility_column,
                    "inference_unit": "downstream_start_within_vae_checkpoint",
                }
            )
    paired = pd.DataFrame(pairs)
    paired.to_csv(out_dir / f"common_state_paired_contrasts{suffix}.csv", index=False)
    return paired


def _plot(rows: pd.DataFrame, out_dir: Path) -> None:
    subset = rows[
        (rows["trust_mode"] == "weight")
        & rows["threshold_robust_eligible"].astype(bool)
        & np.isclose(rows["radius_fraction"].astype(float), 0.5)
        & rows["candidate"].isin(
            [
                "raw_adam_replay",
                "tangent_oracle_rcond1e-5",
                "tangent_oracle_rcond1e-5_nonlinear",
                "jjt_direct_latent_sgd_linear",
                "latent_adam_replay_linear",
                "latent_adam_replay_nonlinear",
                "jjt_plus_normal_rcond1e-5_eps0.25",
                "jjt_plus_normal_rcond1e-5_eps1",
                "jjt_plus_random_normal_placebo_rcond1e-5_eps1",
            ]
        )
    ].copy()
    if subset.empty:
        return
    methods = list(dict.fromkeys(subset["candidate"].astype(str).tolist()))
    labels = list(dict.fromkeys(subset["label"].astype(str).tolist()))
    fig, axes = plt.subplots(len(labels), 2, figsize=(16, max(5, 4.5 * len(labels))), squeeze=False, constrained_layout=True)
    for row_idx, label in enumerate(labels):
        group = subset[subset["label"] == label]
        data = [group[group["candidate"] == method]["full_train_loss_delta"].to_numpy(dtype=np.float64) for method in methods]
        axes[row_idx, 0].boxplot(data, tick_labels=methods, showfliers=True)
        axes[row_idx, 0].axhline(0.0, color="black", lw=1)
        axes[row_idx, 0].tick_params(axis="x", labelrotation=55)
        axes[row_idx, 0].set_title(f"{label}: matched-weight full-train loss delta")
        axes[row_idx, 0].set_ylabel("candidate loss - common-state loss")
        dose = group[group["candidate_family"].isin(["jjt", "normal_access", "normal_placebo"])].copy()
        dose["dose"] = dose["candidate"].map(
            {
                "jjt_direct_latent_sgd_linear": 0.0,
                "jjt_plus_normal_rcond1e-5_eps0.25": 0.25,
                "jjt_plus_normal_rcond1e-5_eps1": 1.0,
                "jjt_plus_random_normal_placebo_rcond1e-5_eps1": 1.0,
            }
        )
        regular = dose[dose["candidate"] != "jjt_plus_random_normal_placebo_rcond1e-5_eps1"]
        means = regular.groupby("dose")["full_train_loss_delta"].mean().sort_index()
        axes[row_idx, 1].plot(means.index, means.values, marker="o", label="task-normal")
        placebo = dose[dose["candidate"] == "jjt_plus_random_normal_placebo_rcond1e-5_eps1"]
        if not placebo.empty:
            axes[row_idx, 1].scatter([1.0], [placebo["full_train_loss_delta"].mean()], marker="x", s=80, label="orthogonal-normal placebo")
        axes[row_idx, 1].axhline(0.0, color="black", lw=1)
        axes[row_idx, 1].set_xlabel("normal operator epsilon / mean nonzero JtJ eigenvalue")
        axes[row_idx, 1].set_ylabel("mean full-train loss delta")
        axes[row_idx, 1].set_title(f"{label}: normal-access intervention")
        axes[row_idx, 1].legend()
    fig.savefig(out_dir / "common_state_rate_overview.png", dpi=180)
    plt.close(fig)


def _expected_candidates(dense_rotation_count: int) -> tuple[set[str], set[str]]:
    candidates = {
        "raw_adam_replay",
        "raw_adam_fresh",
        "jjt_direct_latent_sgd_linear",
        "latent_sgd_nonlinear",
        "latent_adam_replay_linear",
        "latent_adam_replay_nonlinear",
        "latent_adam_fresh_linear",
        "latent_adam_fresh_nonlinear",
        "latent_adam_rotation_identity_linear",
        "latent_adam_rotation_identity_nonlinear",
    }
    for rcond in PROJECTOR_RCONDS:
        tag = _rcond_tag(rcond)
        candidates.update(
            {
                f"tangent_oracle_rcond{tag}",
                f"tangent_oracle_rcond{tag}_nonlinear",
                f"projected_raw_adam_rcond{tag}",
                f"projected_raw_adam_rcond{tag}_nonlinear",
                f"jjt_plus_normal_rcond{tag}_eps0.25",
                f"jjt_plus_random_normal_placebo_rcond{tag}_eps0.25",
                f"jjt_plus_normal_rcond{tag}_eps1",
                f"jjt_plus_random_normal_placebo_rcond{tag}_eps1",
            }
        )
    native = {
        "raw_adam_replay",
        "raw_adam_fresh",
        "latent_adam_replay_linear",
        "latent_adam_replay_nonlinear",
        "latent_adam_fresh_linear",
        "latent_adam_fresh_nonlinear",
        "latent_adam_rotation_identity_linear",
        "latent_adam_rotation_identity_nonlinear",
    }
    for rotation_id in range(int(dense_rotation_count)):
        for suffix in ("linear", "nonlinear"):
            name = f"latent_adam_rotation_dense{rotation_id}_{suffix}"
            candidates.add(name)
            native.add(name)
    return candidates, native


def _expected_row_suffixes(
    dense_rotation_count: int,
    radius_fractions: tuple[float, ...],
) -> set[tuple[str, str, float]]:
    candidates, native = _expected_candidates(dense_rotation_count)
    suffixes = {
        (candidate, trust_mode, float(radius))
        for candidate in candidates
        for trust_mode in ("weight", "function")
        for radius in radius_fractions
    }
    suffixes.update((candidate, "native", 1.0) for candidate in native)
    suffixes.add(("raw_adam_replay", "zero", 0.0))
    return suffixes


STATE_KEYS = ["label", "source_weight_index", "path_step", "draw_id"]
STATE_CONTEXT_COLUMNS = [
    "run_name",
    "start_bank_position",
    "stream_start_index",
    "task_name",
    "tau",
    "raw_lr",
    "latent_lr",
    "update_batch_sha256",
    "eval_batch_sha256",
    "calibration_images_sha256",
    "calibration_indices_sha256",
    "theta_sha256",
    "z_sha256",
    "path_adam_step",
]


def _equal_series(left: pd.Series, right: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
        return pd.Series(
            np.isclose(
                pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ),
            index=left.index,
        )
    sentinel = "__common_state_missing__"
    return left.astype("string").fillna(sentinel).eq(right.astype("string").fillna(sentinel))


def _row_state_integrity(
    rows: pd.DataFrame,
    states: pd.DataFrame,
    *,
    expected_state_key_set: set[tuple[str, int, int, int]],
    start_bank: pd.DataFrame,
    labels: list[str],
) -> dict[str, Any]:
    actual_row_state_key_set = {
        (str(row.label), int(row.source_weight_index), int(row.path_step), int(row.draw_id))
        for row in rows[STATE_KEYS].drop_duplicates().itertuples(index=False)
    }
    actual_state_key_set = {
        (str(row.label), int(row.source_weight_index), int(row.path_step), int(row.draw_id))
        for row in states[STATE_KEYS].drop_duplicates().itertuples(index=False)
    }
    result: dict[str, Any] = {
        "row_state_group_count": int(len(actual_row_state_key_set)),
        "row_missing_expected_state_key_count": int(len(expected_state_key_set - actual_row_state_key_set)),
        "row_unexpected_state_key_count": int(len(actual_row_state_key_set - expected_state_key_set)),
        "row_missing_diagnostic_state_key_count": int(len(actual_state_key_set - actual_row_state_key_set)),
        "row_orphan_state_key_count": int(len(actual_row_state_key_set - actual_state_key_set)),
        "row_state_many_to_one_valid": False,
        "row_state_unmatched_row_count": int(len(rows)),
        "row_state_context_mismatch_row_count": int(len(rows)),
        "row_state_context_mismatch_by_column": {},
        "state_start_context_unmatched_count": int(len(states)),
        "state_start_context_mismatch_row_count": int(len(states)),
        "state_start_context_mismatch_by_column": {},
    }

    context_columns = [column for column in STATE_CONTEXT_COLUMNS if column in rows.columns and column in states.columns]
    if not states.duplicated(STATE_KEYS).any():
        joined = rows.merge(
            states[STATE_KEYS + context_columns],
            on=STATE_KEYS,
            how="left",
            suffixes=("_row", "_state"),
            indicator=True,
            validate="many_to_one",
        )
        matched = joined["_merge"].eq("both")
        mismatch_any = ~matched
        mismatch_by_column: dict[str, int] = {}
        for column in context_columns:
            equal = _equal_series(joined[f"{column}_row"], joined[f"{column}_state"]) & matched
            mismatch_by_column[column] = int((~equal & matched).sum())
            mismatch_any |= ~equal & matched
        result.update(
            {
                "row_state_many_to_one_valid": True,
                "row_state_unmatched_row_count": int((~matched).sum()),
                "row_state_context_mismatch_row_count": int(mismatch_any.sum()),
                "row_state_context_mismatch_by_column": mismatch_by_column,
            }
        )

    bank = start_bank.copy()
    bank["source_weight_index"] = bank["source_weight_index"].astype(int)
    bank["expected_stream_start_index"] = (
        bank["start_index"].astype(int) if "start_index" in bank.columns else bank["start_bank_position"].astype(int)
    )
    expected_context = pd.concat(
        [bank.assign(label=str(label)) for label in labels],
        ignore_index=True,
        sort=False,
    )
    bank_columns = ["start_bank_position", "task_name", "tau"]
    expected_columns = ["start_bank_position", "task_name", "tau"]
    if "stream_start_index" in states.columns:
        bank_columns.append("expected_stream_start_index")
        expected_columns.append("stream_start_index")
    if not expected_context.duplicated(["label", "source_weight_index"]).any():
        state_start = states.merge(
            expected_context[["label", "source_weight_index"] + bank_columns],
            on=["label", "source_weight_index"],
            how="left",
            suffixes=("_state", "_expected"),
            indicator=True,
            validate="many_to_one",
        )
        matched = state_start["_merge"].eq("both")
        mismatch_any = ~matched
        mismatch_by_column = {}
        for expected_column, state_column in zip(bank_columns, expected_columns, strict=True):
            left_name = f"{state_column}_state" if state_column in bank_columns else state_column
            right_name = f"{expected_column}_expected" if expected_column in states.columns else expected_column
            equal = _equal_series(state_start[left_name], state_start[right_name]) & matched
            mismatch_by_column[state_column] = int((~equal & matched).sum())
            mismatch_any |= ~equal & matched
        result.update(
            {
                "state_start_context_unmatched_count": int((~matched).sum()),
                "state_start_context_mismatch_row_count": int(mismatch_any.sum()),
                "state_start_context_mismatch_by_column": mismatch_by_column,
            }
        )
    return result


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")
    run_dirs = [_run_dir(value) for value in args.run]
    labels = [str(value) for value in args.label]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_steps = {int(value) for value in (args.checkpoint_step or [1, 5, 25, 100])}
    trajectory_steps = max(checkpoint_steps)
    radius_values = args.radius_fraction if args.radius_fraction is not None else [0.125, 0.25, 0.5, 1.0]
    radius_fractions = tuple(float(value) for value in radius_values)
    if not math.isclose(float(args.projector_rcond), PRIMARY_PROJECTOR_RCOND, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(
            f"v3 primary --projector-rcond is fixed at {PRIMARY_PROJECTOR_RCOND:g}; got {float(args.projector_rcond):g}"
        )
    _log(
        "startup "
        f"runs={[str(v) for v in run_dirs]} labels={labels} device={args.device} dtype=float32 seed={args.seed} "
        f"trajectory_steps={trajectory_steps} checkpoints={sorted(checkpoint_steps)} batch_draws={args.batch_draws} "
        f"radius_fractions={radius_fractions} projector_rcond={args.projector_rcond} rotations={args.rotation_count} "
        f"jacobian_chunk_size={args.jacobian_chunk_size} output_dir={out_dir}"
    )
    base_cfg = _load_cfg(run_dirs[0], device=str(args.device), downstream_steps=trajectory_steps)
    device = torch.device(base_cfg.device)
    dtype = torch_dtype(base_cfg)
    adam_error = _validate_adam(device)
    _log(f"stage=unit_validation adam_max_abs_error={adam_error:.6g}")
    if adam_error > 2.0e-7:
        raise RuntimeError(f"manual Adam validation failed: max_abs_error={adam_error:.6g}")
    _log(f"stage=load_data data_root={base_cfg.data_root} device={device} dtype={dtype}")
    task_tensors = _load_task_tensors_for_pipeline(base_cfg, device=device, dtype=dtype)
    weights_cpu, weight_records, weight_key, spec = _load_weight_pool(run_dirs[0])
    source_indices = [int(value) for value in args.source_index]
    start_bank = _load_start_bank(Path(args.start_bank_csv).expanduser().resolve(), source_indices)
    if int(args.max_starts) > 0:
        start_bank = start_bank.iloc[: int(args.max_starts)].copy().reset_index(drop=True)
    start_bank.to_csv(out_dir / "common_state_start_bank.csv", index=False)
    request_payload = {
        "protocol_version": "common_state_rate_v3",
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "runs": [str(value) for value in run_dirs],
        "run_checkpoint_sha256": [_sha256_file(value / "vae_checkpoint.pt") for value in run_dirs],
        "run_config_sha256": [_sha256_file(value / "config.json") for value in run_dirs],
        "run_selected_lrs_sha256": [_sha256_file(value / "selected_lrs.csv") for value in run_dirs],
        "run_weight_pool_sha256": [_sha256_file(value / "weight_pool.pt") for value in run_dirs],
        "run_weight_pool_records_sha256": [
            _sha256_file(value / "weight_pool_records.csv") for value in run_dirs
        ],
        "resolved_lrs": [
            {
                "label": label,
                "raw_lr": _selected_lr(run_dir, "raw"),
                "latent_lr": _selected_lr(run_dir, "decoder_latent"),
            }
            for run_dir, label in zip(run_dirs, labels, strict=True)
        ],
        "weight_pool_cache_key": str(weight_key),
        "labels": labels,
        "start_bank_source_sha256": _sha256_file(Path(args.start_bank_csv).expanduser().resolve()),
        "selected_start_rows_sha256": hashlib.sha256(start_bank.to_csv(index=False).encode("utf-8")).hexdigest(),
        "source_indices": [int(value) for value in start_bank["source_weight_index"].tolist()],
        "device": str(args.device),
        "checkpoint_steps": sorted(checkpoint_steps),
        "batch_draws": int(args.batch_draws),
        "radius_fractions": list(radius_fractions),
        "projector_rcond": float(args.projector_rcond),
        "projector_rcond_grid": list(PROJECTOR_RCONDS),
        "trust_scan_scales": list(TRUST_SCAN_SCALES),
        "jacobian_chunk_size": int(args.jacobian_chunk_size),
        "dense_rotation_count": int(args.rotation_count),
        "seed": int(args.seed),
    }
    request_hash = _stable_json_hash(request_payload)
    existing_manifest_path = out_dir / "manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8")) if existing_manifest_path.is_file() else {}
    )
    cache_request_valid = bool(existing_manifest.get("request_hash") == request_hash)
    _log(f"stage=request request_hash={request_hash} cache_request_valid={cache_request_valid}")
    weights_device = weights_cpu.to(device=device, dtype=dtype)
    rotations = [torch.eye(int(base_cfg.latent_dim), device=device, dtype=dtype)] + [
        _random_orthogonal(int(base_cfg.latent_dim), seed=int(args.seed) + 1000003 * (idx + 1), device=device, dtype=dtype)
        for idx in range(int(args.rotation_count))
    ]
    _log(
        f"stage=start_bank starts={len(start_bank)} source_indices={start_bank['source_weight_index'].astype(int).tolist()} "
        f"weight_pool_key={weight_key} rotations_ready={len(rotations)} identity_controls=1 dense_rotations={args.rotation_count}"
    )

    all_rows: list[pd.DataFrame] = []
    all_states: list[pd.DataFrame] = []
    selected_lrs: list[dict[str, Any]] = []
    for run_dir, label in zip(run_dirs, labels, strict=True):
        rows_path = out_dir / f"common_state_rate_rows_{label}.csv"
        states_path = out_dir / f"common_state_state_diagnostics_{label}.csv"
        if rows_path.is_file() and states_path.is_file() and cache_request_valid and not bool(args.force):
            _log(f"cache_hit label={label} rows={rows_path} states={states_path}")
            all_rows.append(pd.read_csv(rows_path))
            all_states.append(pd.read_csv(states_path))
            selected_lrs.append(
                {
                    "label": label,
                    "raw_lr": _selected_lr(run_dir, "raw"),
                    "latent_lr": _selected_lr(run_dir, "decoder_latent"),
                    "run_dir": str(run_dir),
                }
            )
            continue
        if rows_path.is_file() or states_path.is_file():
            _log(f"cache_miss label={label} reason=request_hash_or_force recomputing rows={rows_path} states={states_path}")
        cfg = _load_cfg(run_dir, device=str(args.device), downstream_steps=trajectory_steps)
        run_weights, _records, run_weight_key, run_spec = _load_weight_pool(run_dir)
        if tuple(run_weights.shape) != tuple(weights_cpu.shape) or run_weight_key != weight_key or int(run_spec.dim) != int(spec.dim):
            raise RuntimeError(f"weight pool/spec mismatch for {run_dir}")
        vae, normalizer, _checkpoint = _load_vae(run_dir, cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
        raw_lr = _selected_lr(run_dir, "raw")
        latent_lr = _selected_lr(run_dir, "decoder_latent")
        selected_lrs.append({"label": label, "raw_lr": raw_lr, "latent_lr": latent_lr, "run_dir": str(run_dir)})
        _log(f"stage=probe label={label} raw_lr={raw_lr:g} latent_lr={latent_lr:g} run_dir={run_dir}")
        label_rows: list[dict[str, Any]] = []
        state_rows: list[dict[str, Any]] = []
        progress = make_progress(cfg, total=len(start_bank), desc=f"common-state rate {label}")
        try:
            for index, start_row in start_bank.iterrows():
                rows, states = _run_start(
                    cfg=cfg,
                    vae=vae,
                    normalizer=normalizer,
                    weights_device=weights_device,
                    start_row=start_row,
                    task_tensors=task_tensors,
                    spec=spec,
                    label=label,
                    run_name=run_dir.name,
                    raw_lr=raw_lr,
                    latent_lr=latent_lr,
                    trajectory_steps=trajectory_steps,
                    checkpoint_steps=checkpoint_steps,
                    batch_draws=int(args.batch_draws),
                    radius_fractions=radius_fractions,
                    projector_rcond=float(args.projector_rcond),
                    jacobian_chunk_size=int(args.jacobian_chunk_size),
                    rotations=rotations,
                    seed=int(args.seed),
                )
                label_rows.extend(rows)
                state_rows.extend(states)
                progress.set_postfix({"source": int(start_row["source_weight_index"]), "rows": len(label_rows), "elapsed": f"{time.perf_counter()-started:.1f}s"})
                progress.update(1)
        finally:
            progress.close()
        rows_frame = pd.DataFrame(label_rows)
        states_frame = pd.DataFrame(state_rows)
        rows_frame, states_frame = _propagate_start_singularity(rows_frame, states_frame)
        rows_frame.to_csv(rows_path, index=False)
        states_frame.to_csv(states_path, index=False)
        all_rows.append(rows_frame)
        all_states.append(states_frame)
        _log(f"stage=write_label label={label} rows={len(rows_frame)} states={len(states_frame)} rows_path={rows_path}")

    rows = pd.concat(all_rows, ignore_index=True, sort=False)
    states = pd.concat(all_states, ignore_index=True, sort=False)
    rows.to_csv(out_dir / "common_state_rate_rows.csv", index=False)
    states.to_csv(out_dir / "common_state_state_diagnostics.csv", index=False)
    pd.DataFrame(selected_lrs).to_csv(out_dir / "common_state_selected_lrs.csv", index=False)
    paired = _summaries(rows, out_dir, eligibility_column="threshold_robust_eligible")
    paired_threshold_specific = _summaries(
        rows,
        out_dir,
        eligibility_column="threshold_specific_eligible",
        suffix="_threshold_specific",
        write_coverage=False,
    )
    _plot(rows, out_dir)
    expected_candidates, expected_native_candidates = _expected_candidates(int(args.rotation_count))
    state_keys = STATE_KEYS
    expected_states = int(len(labels) * len(start_bank) * len(checkpoint_steps) * int(args.batch_draws))
    expected_rows_per_state = int(
        len(expected_candidates) * 2 * len(radius_fractions) + len(expected_native_candidates) + 1
    )
    expected_rows = int(expected_states * expected_rows_per_state)
    rows_per_state = rows.groupby(state_keys).size()
    duplicate_keys = state_keys + ["candidate", "trust_mode", "radius_fraction"]
    expected_state_key_set = {
        (str(label), int(source), int(path_step), int(draw_id))
        for label in labels
        for source in start_bank["source_weight_index"].astype(int).tolist()
        for path_step in checkpoint_steps
        for draw_id in range(int(args.batch_draws))
    }
    actual_state_key_set = {
        (str(row.label), int(row.source_weight_index), int(row.path_step), int(row.draw_id))
        for row in states[state_keys].itertuples(index=False)
    }
    row_state_integrity = _row_state_integrity(
        rows,
        states,
        expected_state_key_set=expected_state_key_set,
        start_bank=start_bank,
        labels=labels,
    )
    expected_suffixes = _expected_row_suffixes(int(args.rotation_count), radius_fractions)
    bad_row_grid_states = 0
    for _state_key, group in rows.groupby(state_keys):
        actual_suffixes = {
            (str(row.candidate), str(row.trust_mode), float(row.radius_fraction))
            for row in group[["candidate", "trust_mode", "radius_fraction"]].itertuples(index=False)
        }
        bad_row_grid_states += int(actual_suffixes != expected_suffixes)
    zero_rows = rows[rows["trust_mode"] == "zero"].copy()
    identity_keys = state_keys + ["trust_mode", "radius_fraction"]
    identity_columns = [
        "candidate_scale",
        "matched_trust",
        "trust_match_rel_error",
        "delta_weight_norm",
        "delta_function_rms",
        "update_batch_loss_delta",
        "eval_batch_loss_delta",
        "full_train_loss_delta",
        "test_loss_delta",
        "same_scale_linear_full_train_loss_delta",
        "same_scale_linear_test_loss_delta",
    ]
    identity_joins: list[pd.DataFrame] = []
    for suffix in ("linear", "nonlinear"):
        identity_base = rows[rows["candidate"] == f"latent_adam_replay_{suffix}"][identity_keys + identity_columns]
        identity_rotation = rows[rows["candidate"] == f"latent_adam_rotation_identity_{suffix}"][
            identity_keys + identity_columns
        ]
        identity_joins.append(
            identity_base.merge(identity_rotation, on=identity_keys, suffixes=("_base", "_identity"))
        )
    identity_join = pd.concat(identity_joins, ignore_index=True, sort=False)
    identity_max_abs = 0.0
    for column in identity_columns:
        if not identity_join.empty:
            differences = (
                pd.to_numeric(identity_join[f"{column}_base"], errors="coerce")
                - pd.to_numeric(identity_join[f"{column}_identity"], errors="coerce")
            ).abs().dropna()
            if not differences.empty:
                identity_max_abs = max(identity_max_abs, float(differences.max()))
    trust_rows = rows[rows["trust_mode"].isin(["weight", "function"])]
    all_finite_columns = [
        "delta_weight_norm",
        "delta_function_rms",
        "update_batch_loss_delta",
        "eval_batch_loss_delta",
        "full_train_loss_delta",
        "test_loss_delta",
        "same_scale_linear_update_batch_loss_delta",
        "same_scale_linear_eval_batch_loss_delta",
        "same_scale_linear_full_train_loss_delta",
        "same_scale_linear_test_loss_delta",
    ]
    validation = {
        "request_hash": request_hash,
        "adam_max_abs_error": float(adam_error),
        "all_finite_core": bool(np.isfinite(rows[all_finite_columns].to_numpy(dtype=np.float64)).all()),
        "rows": int(len(rows)),
        "expected_rows": expected_rows,
        "rows_per_state_min": int(rows_per_state.min()),
        "rows_per_state_max": int(rows_per_state.max()),
        "expected_rows_per_state": expected_rows_per_state,
        "states": int(len(states)),
        "expected_states": expected_states,
        "labels": sorted(rows["label"].astype(str).unique().tolist()),
        "expected_labels": sorted(labels),
        "starts_per_label": {str(k): int(v) for k, v in rows.groupby("label")["source_weight_index"].nunique().items()},
        "path_steps": sorted(int(v) for v in rows["path_step"].unique().tolist()),
        "expected_path_steps": sorted(checkpoint_steps),
        "batch_draws": sorted(int(v) for v in rows["draw_id"].unique().tolist()),
        "expected_batch_draws": list(range(int(args.batch_draws))),
        "candidate_count": int(rows["candidate"].nunique()),
        "expected_candidate_count": int(len(expected_candidates)),
        "candidate_set_matches": bool(set(rows["candidate"].astype(str).unique()) == expected_candidates),
        "duplicate_row_keys": int(rows.duplicated(duplicate_keys).sum()),
        "duplicate_state_keys": int(states.duplicated(state_keys).sum()),
        "missing_state_key_count": int(len(expected_state_key_set - actual_state_key_set)),
        "unexpected_state_key_count": int(len(actual_state_key_set - expected_state_key_set)),
        "bad_row_grid_state_count": int(bad_row_grid_states),
        "request_payload_rehash_matches": bool(_stable_json_hash(request_payload) == request_hash),
        "theta_hash_max_unique_across_draws": int(states.groupby(["label", "source_weight_index", "path_step"])["theta_sha256"].nunique().max()),
        "update_batch_hash_min_unique_across_draws": int(states.groupby(["label", "source_weight_index", "path_step"])["update_batch_sha256"].nunique().min()),
        "eval_batch_hash_min_unique_across_draws": int(states.groupby(["label", "source_weight_index", "path_step"])["eval_batch_sha256"].nunique().min()),
        "calibration_hash_max_unique_per_start": int(states.groupby(["label", "source_weight_index"])["calibration_indices_sha256"].nunique().max()),
        "update_eval_index_overlap_max": int(states["update_eval_index_overlap"].max()),
        "update_calibration_index_overlap_max": int(states["update_calibration_index_overlap"].max()),
        "eval_calibration_index_overlap_max": int(states["eval_calibration_index_overlap"].max()),
        "max_weight_match_rel_error": float(rows.loc[rows["trust_mode"] == "weight", "trust_match_rel_error"].max()),
        "max_function_match_rel_error": float(rows.loc[rows["trust_mode"] == "function", "trust_match_rel_error"].max()),
        "trust_response_monotone_all": bool(trust_rows["trust_response_monotone"].astype(bool).all()),
        "trust_scale_boundary_hit_count": int(trust_rows["trust_scale_boundary_hit"].astype(bool).sum()),
        "trust_bracket_missing_count": int((~trust_rows["trust_bracket_found"].astype(bool)).sum()),
        "trust_local_nonpositive_count": int((~trust_rows["trust_local_positive_slope"].astype(bool)).sum()),
        "trust_eligible_fraction": float(trust_rows["trust_eligible"].astype(bool).mean()),
        "technical_eligible_fraction": float(trust_rows["technical_eligible"].astype(bool).mean()),
        "regular_stratum_eligible_fraction": float(trust_rows["regular_stratum_eligible"].astype(bool).mean()),
        "threshold_specific_eligible_fraction": float(
            trust_rows["threshold_specific_eligible"].astype(bool).mean()
        ),
        "threshold_robust_eligible_fraction": float(
            trust_rows["threshold_robust_eligible"].astype(bool).mean()
        ),
        "singularity_start_count": int(states.loc[states["singularity_stratum"].astype(bool), "source_weight_index"].nunique()),
        "singularity_sources": sorted(
            int(value)
            for value in states.loc[states["singularity_stratum"].astype(bool), "source_weight_index"].unique().tolist()
        ),
        "tangent_pseudoinverse_stable_fraction": float(states["tangent_pseudoinverse_stable"].astype(bool).mean()),
        "projected_pseudoinverse_stable_fraction": float(states["projected_pseudoinverse_stable"].astype(bool).mean()),
        "normal_operator_stable_fraction": float(states["normal_operator_stable"].astype(bool).mean()),
        "nonlocal_retraction_row_count": int((~rows["retraction_executable"].astype(bool)).sum()),
        "projector_idempotence_rel_max": float(states["projector_idempotence_rel"].max()),
        "projector_normal_jt_rel_max": float(states["projector_normal_jt_rel"].max()),
        "random_normal_task_normal_cos_max_abs": float(states["random_normal_task_normal_cos"].abs().max()),
        "projector_rank_stable_fraction": float(
            (
                (states["projector_rank_rcond_x0.1"] == states["projector_rank_rcond_x1"])
                & (states["projector_rank_rcond_x1"] == states["projector_rank_rcond_x10"])
            ).mean()
        ),
        "zero_row_count": int(len(zero_rows)),
        "expected_zero_row_count": expected_states,
        "zero_delta_weight_norm_max": float(zero_rows["delta_weight_norm"].abs().max()),
        "zero_loss_delta_max_abs": float(
            zero_rows[["update_batch_loss_delta", "eval_batch_loss_delta", "full_train_loss_delta", "test_loss_delta"]]
            .abs()
            .to_numpy(dtype=np.float64)
            .max()
        ),
        "identity_rotation_join_rows": int(len(identity_join)),
        "identity_rotation_expected_rows": int(
            2 * expected_states * (2 * len(radius_fractions) + 1)
        ),
        "identity_rotation_max_abs_metric_delta": float(identity_max_abs),
        "paired_contrast_rows": int(len(paired)),
        "paired_threshold_specific_contrast_rows": int(len(paired_threshold_specific)),
        **row_state_integrity,
    }
    acceptance_checks = {
        "adam_parity": validation["adam_max_abs_error"] <= 2.0e-7,
        "all_finite": validation["all_finite_core"],
        "complete_rows": validation["rows"] == validation["expected_rows"],
        "complete_states": validation["states"] == validation["expected_states"],
        "rows_per_state": validation["rows_per_state_min"] == expected_rows_per_state == validation["rows_per_state_max"],
        "labels": validation["labels"] == validation["expected_labels"],
        "path_steps": validation["path_steps"] == validation["expected_path_steps"],
        "batch_draws": validation["batch_draws"] == validation["expected_batch_draws"],
        "candidates": validation["candidate_set_matches"],
        "no_duplicates": validation["duplicate_row_keys"] == 0,
        "no_duplicate_states": validation["duplicate_state_keys"] == 0,
        "exact_state_grid": (
            validation["missing_state_key_count"] == 0
            and validation["unexpected_state_key_count"] == 0
        ),
        "exact_row_grid": (
            validation["bad_row_grid_state_count"] == 0
            and validation["row_state_group_count"] == expected_states
            and validation["row_missing_expected_state_key_count"] == 0
            and validation["row_unexpected_state_key_count"] == 0
            and validation["row_missing_diagnostic_state_key_count"] == 0
            and validation["row_orphan_state_key_count"] == 0
        ),
        "row_state_linkage": (
            validation["row_state_many_to_one_valid"]
            and validation["row_state_unmatched_row_count"] == 0
            and validation["row_state_context_mismatch_row_count"] == 0
        ),
        "state_start_context": (
            validation["state_start_context_unmatched_count"] == 0
            and validation["state_start_context_mismatch_row_count"] == 0
        ),
        "request_hash_integrity": validation["request_payload_rehash_matches"],
        "common_theta_across_draws": validation["theta_hash_max_unique_across_draws"] == 1,
        "draw_batches_unique": (
            validation["update_batch_hash_min_unique_across_draws"] == int(args.batch_draws)
            and validation["eval_batch_hash_min_unique_across_draws"] == int(args.batch_draws)
        ),
        "calibration_fixed_per_start": validation["calibration_hash_max_unique_per_start"] == 1,
        "batch_calibration_disjoint": (
            validation["update_eval_index_overlap_max"] == 0
            and validation["update_calibration_index_overlap_max"] == 0
            and validation["eval_calibration_index_overlap_max"] == 0
        ),
        "projector_idempotence": validation["projector_idempotence_rel_max"] <= 1.0e-5,
        "projector_normal": validation["projector_normal_jt_rel_max"] <= 1.0e-5,
        "normal_placebo_orthogonal": validation["random_normal_task_normal_cos_max_abs"] <= 1.0e-5,
        "zero_arm": (
            validation["zero_row_count"] == validation["expected_zero_row_count"]
            and validation["zero_delta_weight_norm_max"] <= 1.0e-12
            and validation["zero_loss_delta_max_abs"] <= 1.0e-12
        ),
        "identity_rotation": (
            validation["identity_rotation_join_rows"] == validation["identity_rotation_expected_rows"]
            and validation["identity_rotation_max_abs_metric_delta"] <= 1.0e-6
        ),
        "contrasts_written": validation["paired_contrast_rows"] > 0,
    }
    validation["acceptance_checks"] = acceptance_checks
    validation["acceptance_pass"] = bool(all(acceptance_checks.values()))
    (out_dir / "common_state_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "runs": [str(value) for value in run_dirs],
        "labels": labels,
        "output_dir": str(out_dir),
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(args.seed),
        "start_bank_csv": str(Path(args.start_bank_csv).expanduser().resolve()),
        "source_indices": [int(value) for value in start_bank["source_weight_index"].tolist()],
        "checkpoint_steps": sorted(checkpoint_steps),
        "batch_draws": int(args.batch_draws),
        "radius_fractions": list(radius_fractions),
        "projector_rcond": float(args.projector_rcond),
        "projector_rcond_grid": list(PROJECTOR_RCONDS),
        "trust_scan_scales": list(TRUST_SCAN_SCALES),
        "jacobian_chunk_size": int(args.jacobian_chunk_size),
        "rotation_count": int(args.rotation_count),
        "rotation_identity_controls": 1,
        "request_hash": request_hash,
        "request": request_payload,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(
        "done "
        f"elapsed_sec={time.perf_counter()-started:.2f} acceptance_pass={validation['acceptance_pass']} "
        f"rows={out_dir / 'common_state_rate_rows.csv'} summary={out_dir / 'common_state_rate_summary.csv'} "
        f"contrasts={out_dir / 'common_state_paired_contrasts.csv'} validation={out_dir / 'common_state_validation.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Common-state decomposition of raw versus VAE-latent local convergence rate.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-bank-csv", required=True)
    parser.add_argument("--source-index", action="append", type=int, default=[])
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-step", action="append", type=int, default=None)
    parser.add_argument("--batch-draws", type=int, default=2)
    parser.add_argument("--radius-fraction", action="append", type=float, default=None)
    parser.add_argument("--projector-rcond", type=float, default=PRIMARY_PROJECTOR_RCOND)
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--rotation-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
