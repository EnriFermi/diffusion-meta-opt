from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    decode_weights,
    decoder_jacobians,
    encode_weights,
    load_torch_cache,
    logits_from_flat,
    spec_from_payload,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline import (
    _load_task_tensors_for_pipeline,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.progress import make_progress


ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing").resolve()


def _log(message: str) -> None:
    print(f"[trajectory_realization] {message}", flush=True)


def _run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / value).resolve()


def _load_cfg(run_dir: Path, *, device: str, downstream_steps: int) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{run_dir / 'config.json'} does not contain a config mapping")
    cfg = ExperimentConfig(**raw_cfg)
    return replace(
        cfg,
        device=str(device),
        dtype="float32",
        show_progress=True,
        progress_backend="text",
        downstream_steps=int(downstream_steps),
    )


def _selected_lr(run_dir: Path, method: str) -> float:
    rows = pd.read_csv(run_dir / "selected_lrs.csv")
    sub = rows[
        (rows["method"].astype(str) == str(method))
        & (pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int) == 1)
    ]
    if sub.empty:
        raise RuntimeError(f"no selected LR for method={method!r} in {run_dir / 'selected_lrs.csv'}")
    return float(sub.iloc[0]["candidate_lr"])


def _load_weight_pool(run_dir: Path) -> tuple[torch.Tensor, pd.DataFrame, str, Any]:
    payload = load_torch_cache(run_dir / "weight_pool.pt")
    if payload is None or not isinstance(payload.get("weights"), torch.Tensor):
        raise RuntimeError(f"could not load weight_pool.pt from {run_dir}")
    records = pd.DataFrame(payload.get("records", []))
    if records.empty:
        records_path = run_dir / "weight_pool_records.csv"
        if not records_path.is_file():
            raise FileNotFoundError(records_path)
        records = pd.read_csv(records_path)
    return payload["weights"].detach().cpu(), records, str(payload.get("cache_key", "")), spec_from_payload(payload["spec"])


def _load_vae(run_dir: Path, cfg: ExperimentConfig, weight_dim: int, *, device: torch.device, dtype: torch.dtype):
    payload = load_torch_cache(run_dir / "vae_checkpoint.pt")
    if payload is None:
        raise RuntimeError(f"could not load vae_checkpoint.pt from {run_dir}")
    normalizer = WeightNormalizer.from_state_dict(payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weight_dim))
    vae.load_state_dict(payload["model_state"])
    vae.to(device=device, dtype=dtype).eval()
    return vae, normalizer, payload


def _task_tensor_set(task_tensors: dict[str, Any], task_name: str):
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found")


def _batch_indices(task_set: Any, *, batch_size: int, step: int, start_bank_position: int) -> torch.Tensor | None:
    train_count = int(task_set.train_labels.shape[0])
    if int(batch_size) <= 0 or int(batch_size) >= train_count:
        return None
    offset = (int(start_bank_position) * 1009 + int(step) * int(batch_size)) % train_count
    return (torch.arange(int(batch_size), device=task_set.train_labels.device) + offset).remainder(train_count).long()


def _loss_acc(
    flat: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    split: str,
    tau: float,
    batch_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        images, labels = task_set.train_images, task_set.train_labels
        if batch_indices is not None:
            images = images.index_select(0, batch_indices)
            labels = labels.index_select(0, batch_indices)
    elif split == "test":
        images, labels = task_set.test_images, task_set.test_labels
    else:
        raise ValueError(f"unknown split={split!r}")
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


def _finite(value: torch.Tensor | float, *, penalty: float = 1.0e6) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    value = float(value)
    return value if math.isfinite(value) else float(penalty)


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(float(den), 1.0e-30))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().double().flatten()
    b64 = b.detach().double().flatten()
    denom = a64.norm() * b64.norm()
    if float(denom.detach().cpu().item()) <= 1.0e-30:
        return float("nan")
    return float((torch.dot(a64, b64) / denom).detach().cpu().item())


def _tensor_sha256(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "full_batch"
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _probe_seed(base_seed: int, *, source_weight_index: int, step: int, probe_id: int) -> int:
    value = (
        int(base_seed)
        + 1_000_003 * int(source_weight_index)
        + 9_176 * int(step)
        + 10_000_019 * int(probe_id)
    )
    return int(value % (2**31 - 1))


def _x_probe(dim: int, *, seed: int, device: torch.device, dtype: torch.dtype, kind: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    kind_value = str(kind).strip().lower()
    if kind_value in {"rademacher", "rad", "sign"}:
        probe = torch.randint(0, 2, (int(dim),), generator=generator, device="cpu", dtype=torch.int64)
        probe = probe.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    elif kind_value in {"gaussian", "normal"}:
        probe = torch.randn((int(dim),), generator=generator, device="cpu", dtype=torch.float32)
        probe = probe * (math.sqrt(float(dim)) / probe.norm().clamp_min(1.0e-12))
    else:
        raise ValueError(f"unsupported probe kind {kind!r}")
    return probe.to(device=device, dtype=dtype)


def _hvp_x(
    theta: torch.Tensor,
    v: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    theta_req = theta.detach().clone().requires_grad_(True)
    loss, _ = _loss_acc(theta_req, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    grad = torch.autograd.grad(loss, theta_req, create_graph=True, retain_graph=True)[0]
    hvp = torch.autograd.grad((grad * v.detach()).sum(), theta_req, create_graph=False, retain_graph=False)[0]
    return hvp.detach()


def _cg_solve_full_j(
    *,
    jacobian64: torch.Tensor,
    v: torch.Tensor,
    lambda_abs: float,
    max_iters: int,
    tol: float,
    init: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    v64 = v.detach().double()
    lam = float(lambda_abs)

    def matvec(q64: torch.Tensor) -> torch.Tensor:
        return jacobian64 @ (jacobian64.T @ q64) + lam * q64

    if str(init).strip().lower() in {"lambda", "v_over_lambda", "null"}:
        w = v64 / lam
    else:
        w = torch.zeros_like(v64)
    aw = matvec(w)
    r = v64 - aw
    p = r.clone()
    v_norm = float(v64.norm().detach().cpu().item())
    initial_rel = float(r.norm().detach().cpu().item()) / max(v_norm, 1.0e-30)
    rel = initial_rel
    rs_old = torch.dot(r, r)
    residual_history = [float(rel)]
    converged = bool(rel <= float(tol))
    breakdown = ""
    iters = 0
    for idx in range(int(max_iters)):
        if converged:
            break
        ap = matvec(p)
        denom = torch.dot(p, ap)
        denom_value = float(denom.detach().cpu().item())
        if (not math.isfinite(denom_value)) or denom_value <= 0.0:
            breakdown = f"non_positive_denom:{denom_value:.6g}"
            break
        alpha = rs_old / denom
        w = w + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        rel = math.sqrt(max(0.0, float(rs_new.detach().cpu().item()))) / max(v_norm, 1.0e-30)
        residual_history.append(float(rel))
        iters = idx + 1
        if rel <= float(tol):
            converged = True
            rs_old = rs_new
            break
        beta = rs_new / rs_old.clamp_min(1.0e-300)
        p = r + beta * p
        rs_old = rs_new
    final_residual = matvec(w) - v64
    final_rel = float(final_residual.norm().detach().cpu().item()) / max(v_norm, 1.0e-30)
    return {
        "w": w.detach(),
        "cg_iterations": int(iters),
        "cg_converged": bool(converged),
        "cg_breakdown": breakdown,
        "cg_initial_rel_residual": float(initial_rel),
        "cg_final_rel_residual": float(final_rel),
        "cg_elapsed_sec": float(time.perf_counter() - start),
        "cg_residual_history_json": json.dumps(residual_history),
        "vTw": float(torch.dot(v64, w).detach().cpu().item()),
    }


def _fresh_raw_adam_step(
    *,
    theta: torch.Tensor,
    raw_lr: float,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    value = theta.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([value], lr=float(raw_lr))
    opt.zero_grad(set_to_none=True)
    loss, _ = _loss_acc(value, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    loss.backward()
    opt.step()
    return value.detach() - theta.detach()


def _fresh_latent_adam_step(
    *,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    z: torch.Tensor,
    latent_lr: float,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    value = z.detach().clone().requires_grad_(True)
    theta0 = decode_weights(vae, normalizer, value.reshape(1, -1)).squeeze(0).detach()
    opt = torch.optim.Adam([value], lr=float(latent_lr))
    opt.zero_grad(set_to_none=True)
    theta = decode_weights(vae, normalizer, value.reshape(1, -1)).squeeze(0)
    loss, _ = _loss_acc(theta, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    loss.backward()
    opt.step()
    theta1 = decode_weights(vae, normalizer, value.reshape(1, -1)).squeeze(0).detach()
    return theta1 - theta0


def _adam_delta_from_grad(
    *,
    grad: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    grad_detached = grad.detach()
    if "exp_avg" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad_detached)
        state["exp_avg_sq"] = torch.zeros_like(grad_detached)
    state["step"] = int(state["step"]) + 1
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    exp_avg.mul_(float(beta1)).add_(grad_detached, alpha=1.0 - float(beta1))
    exp_avg_sq.mul_(float(beta2)).addcmul_(grad_detached, grad_detached, value=1.0 - float(beta2))
    bias_correction1 = 1.0 - float(beta1) ** int(state["step"])
    bias_correction2 = 1.0 - float(beta2) ** int(state["step"])
    denom = exp_avg_sq.sqrt().div(math.sqrt(max(bias_correction2, 1.0e-30))).add(float(eps))
    return exp_avg.div(denom).mul(-float(lr) / max(bias_correction1, 1.0e-30)).detach()


def _project_x_delta_to_decoder_tangent(
    *,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    z: torch.Tensor,
    delta_theta: torch.Tensor,
    damping_rel: float,
    jacobian_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    jacobian = decoder_jacobians(
        vae,
        normalizer,
        None,
        z.detach().reshape(1, -1),
        create_graph=False,
        chunk_size=int(jacobian_chunk_size),
    ).squeeze(0).detach()
    j64 = jacobian.double()
    metric = j64.T @ j64
    trace = float(torch.trace(metric).detach().cpu().item())
    trace_per_dim = trace / max(float(metric.shape[0]), 1.0)
    damping_abs = float(damping_rel) * max(trace_per_dim, 1.0e-30)
    eye = torch.eye(int(metric.shape[0]), device=metric.device, dtype=torch.float64)
    dz64 = torch.linalg.solve(metric + damping_abs * eye, j64.T @ delta_theta.detach().double())
    projected_linear = j64 @ dz64
    residual = delta_theta.detach().double() - projected_linear
    raw_norm = float(delta_theta.detach().float().norm().cpu().item())
    projected_norm = float(projected_linear.norm().detach().cpu().item())
    return dz64.to(device=z.device, dtype=z.dtype).detach(), {
        "projected_raw_adam_damping_abs": float(damping_abs),
        "projected_raw_adam_delta_norm": raw_norm,
        "projected_raw_adam_linear_norm": projected_norm,
        "projected_raw_adam_projection_cos": _cosine(projected_linear, delta_theta),
        "projected_raw_adam_projection_norm_fraction": _safe_ratio(projected_norm, raw_norm),
        "projected_raw_adam_projection_residual_fraction": _safe_ratio(float(residual.norm().detach().cpu().item()), raw_norm),
        "projected_raw_adam_dz_norm": float(dz64.norm().detach().cpu().item()),
    }


def _geometry_and_cg_rows(
    *,
    cfg: ExperimentConfig,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    z: torch.Tensor,
    theta: torch.Tensor,
    w0: torch.Tensor,
    theta_dec0: torch.Tensor,
    label: str,
    run_name: str,
    method: str,
    start_row: pd.Series,
    step: int,
    raw_lr: float,
    latent_lr: float,
    task_set: Any,
    spec: Any,
    tau: float,
    batch_indices: torch.Tensor | None,
    lambda_abs: float,
    tangent_damping_rel: float,
    probe_kind: str,
    probes: int,
    seed: int,
    cg_max_iters: int,
    cg_tol: float,
    cg_init: str,
    jacobian_chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_weight_index = int(start_row["source_weight_index"])
    start_bank_position = int(start_row["start_bank_position"])
    theta_req = theta.detach().clone().requires_grad_(True)
    batch_loss, _ = _loss_acc(theta_req, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    grad_theta = torch.autograd.grad(batch_loss, theta_req, retain_graph=False, create_graph=False)[0].detach()
    neg_grad = -grad_theta
    grad_norm = _norm(grad_theta)

    jacobian = decoder_jacobians(
        vae,
        normalizer,
        None,
        z.detach().reshape(1, -1),
        create_graph=False,
        chunk_size=int(jacobian_chunk_size),
    ).squeeze(0).detach()
    j64 = jacobian.double()
    metric = j64.T @ j64
    eig = torch.linalg.eigvalsh(metric.float()).detach()
    eig_min = float(eig.min().detach().cpu().item())
    eig_max = float(eig.max().detach().cpu().item())
    trace = float(torch.trace(metric).detach().cpu().item())
    trace2 = float((metric * metric).sum().detach().cpu().item())
    latent_dim = int(metric.shape[0])
    trace_per_dim = trace / max(float(latent_dim), 1.0)
    effective_rank = trace * trace / max(trace2, 1.0e-30)
    tangent_damping_abs = float(tangent_damping_rel) * max(trace_per_dim, 1.0e-30)
    eye = torch.eye(latent_dim, device=metric.device, dtype=torch.float64)
    rhs = -(j64.T @ grad_theta.detach().double())
    natural_dz64 = torch.linalg.solve(metric + tangent_damping_abs * eye, rhs)
    natural_linear = j64 @ natural_dz64
    residual = neg_grad.detach().double() - natural_linear
    fresh_raw_step = _fresh_raw_adam_step(
        theta=theta,
        raw_lr=float(raw_lr),
        task_set=task_set,
        spec=spec,
        tau=tau,
        batch_indices=batch_indices,
    )
    fresh_latent_step = _fresh_latent_adam_step(
        vae=vae,
        normalizer=normalizer,
        z=z,
        latent_lr=float(latent_lr),
        task_set=task_set,
        spec=spec,
        tau=tau,
        batch_indices=batch_indices,
    )
    raw_projected_dz64 = torch.linalg.solve(metric + tangent_damping_abs * eye, j64.T @ fresh_raw_step.detach().double())
    raw_projected_linear = j64 @ raw_projected_dz64
    raw_projection_residual = fresh_raw_step.detach().double() - raw_projected_linear

    common = {
        "label": label,
        "run_name": run_name,
        "method": str(method),
        "source_weight_index": source_weight_index,
        "start_bank_position": start_bank_position,
        "task_name": str(start_row["task_name"]),
        "tau": float(tau),
        "step": int(step),
        "raw_lr": float(raw_lr),
        "latent_lr": float(latent_lr),
        "batch_indices_sha256": _tensor_sha256(batch_indices),
        "batch_size": int(task_set.train_labels.shape[0] if batch_indices is None else batch_indices.numel()),
        "theta_norm": _norm(theta),
        "theta_rel_w0": _safe_ratio(_norm(theta - w0), _norm(w0)),
        "theta_rel_dec0": _safe_ratio(_norm(theta - theta_dec0), _norm(w0)),
        "decoded_start_reconstruction_rel_l2": _safe_ratio(_norm(theta_dec0 - w0), _norm(w0)),
        "batch_loss_for_geometry": _finite(batch_loss, penalty=float(cfg.finite_penalty)),
        "grad_theta_norm": grad_norm,
        "trace_jtj": trace,
        "trace_jtj_per_dim": trace_per_dim,
        "metric_effective_rank": effective_rank,
        "metric_eig_min": eig_min,
        "metric_eig_max": eig_max,
        "metric_condition": eig_max / max(eig_min, 1.0e-30),
        "tangent_damping_rel": float(tangent_damping_rel),
        "tangent_damping_abs": tangent_damping_abs,
        "projection_cos_with_neg_grad": _cosine(natural_linear, neg_grad),
        "projection_norm_fraction": _safe_ratio(float(natural_linear.norm().detach().cpu().item()), grad_norm),
        "normal_residual_fraction": _safe_ratio(float(residual.norm().detach().cpu().item()), grad_norm),
        "fresh_raw_adam_step_norm": _norm(fresh_raw_step),
        "fresh_latent_adam_step_norm": _norm(fresh_latent_step),
        "fresh_raw_adam_cos_with_neg_grad": _cosine(fresh_raw_step, neg_grad),
        "fresh_latent_adam_cos_with_neg_grad": _cosine(fresh_latent_step, neg_grad),
        "fresh_latent_adam_cos_with_natural_linear": _cosine(fresh_latent_step, natural_linear),
        "fresh_raw_adam_projection_cos": _cosine(raw_projected_linear, fresh_raw_step),
        "fresh_raw_adam_projection_norm_fraction": _safe_ratio(float(raw_projected_linear.norm().detach().cpu().item()), _norm(fresh_raw_step)),
        "fresh_raw_adam_projection_residual_fraction": _safe_ratio(float(raw_projection_residual.norm().detach().cpu().item()), _norm(fresh_raw_step)),
    }

    cg_rows: list[dict[str, Any]] = []
    for probe_id in range(int(probes)):
        probe = _x_probe(
            int(spec.dim),
            seed=_probe_seed(int(seed), source_weight_index=source_weight_index, step=int(step), probe_id=int(probe_id)),
            device=theta.device,
            dtype=theta.dtype,
            kind=str(probe_kind),
        )
        hvp = _hvp_x(theta, probe, task_set=task_set, spec=spec, tau=tau, batch_indices=batch_indices)
        cg = _cg_solve_full_j(
            jacobian64=j64,
            v=probe,
            lambda_abs=float(lambda_abs),
            max_iters=int(cg_max_iters),
            tol=float(cg_tol),
            init=str(cg_init),
        )
        w = cg.pop("w")
        jt_h = j64.T @ hvp.detach().double()
        c3_forward = float((jt_h * jt_h).sum().detach().cpu().item())
        probe_norm2 = float(probe.float().square().sum().detach().cpu().item())
        cg_rows.append(
            {
                **common,
                "probe_id": int(probe_id),
                "probe_kind": str(probe_kind),
                "probe_sha256": _tensor_sha256(probe),
                "lambda_abs": float(lambda_abs),
                "lambda_rel_to_trace_jtj": float(lambda_abs / max(trace_per_dim, 1.0e-30)),
                "probe_norm2": probe_norm2,
                "hvp_norm2": float(hvp.float().square().sum().detach().cpu().item()),
                "c3_forward": c3_forward,
                "c3_inverse": float(cg["vTw"]),
                "c3_total": float(c3_forward + float(cg["vTw"])),
                "lambda_vtw_ratio": float(float(lambda_abs) * float(cg["vTw"]) / max(probe_norm2, 1.0e-30)),
                **cg,
            }
        )
    return common, cg_rows


def _eval_path_state(
    *,
    theta: torch.Tensor,
    task_set: Any,
    spec: Any,
    tau: float,
    penalty: float,
) -> dict[str, float]:
    with torch.no_grad():
        train_loss, train_acc = _loss_acc(theta.detach(), task_set=task_set, spec=spec, split="train", tau=tau)
        test_loss, test_acc = _loss_acc(theta.detach(), task_set=task_set, spec=spec, split="test", tau=tau)
    return {
        "train_loss": _finite(train_loss, penalty=penalty),
        "train_acc": float(train_acc.detach().cpu().item()),
        "test_loss": _finite(test_loss, penalty=penalty),
        "test_acc": float(test_acc.detach().cpu().item()),
    }


def _run_start_paths(
    *,
    cfg: ExperimentConfig,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    weights_device: torch.Tensor,
    start_row: pd.Series,
    task_tensors: dict[str, Any],
    spec: Any,
    label: str,
    run_name: str,
    raw_lr: float,
    latent_lr: float,
    steps: int,
    checkpoint_steps: set[int],
    lambda_abs: float,
    tangent_damping_rel: float,
    probe_kind: str,
    probes: int,
    seed: int,
    cg_max_iters: int,
    cg_tol: float,
    cg_init: str,
    jacobian_chunk_size: int,
    include_projected_raw_adam_tangent: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_weight_index = int(start_row["source_weight_index"])
    start_bank_position = int(start_row["start_bank_position"])
    task_name = str(start_row["task_name"])
    tau = float(start_row["tau"])
    task_set = _task_tensor_set(task_tensors, task_name)
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))
    w0 = weights_device[source_weight_index].detach()
    w0_norm = _norm(w0)
    with torch.no_grad():
        z0 = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
        theta_dec0 = decode_weights(vae, normalizer, z0.reshape(1, -1)).squeeze(0).detach()

    raw_value = w0.detach().clone().requires_grad_(True)
    decoded_raw_value = theta_dec0.detach().clone().requires_grad_(True)
    latent_value = z0.detach().clone().requires_grad_(True)
    projected_latent_value = z0.detach().clone()
    projected_adam_state: dict[str, Any] = {}
    raw_opt = torch.optim.Adam([raw_value], lr=float(raw_lr))
    decoded_raw_opt = torch.optim.Adam([decoded_raw_value], lr=float(raw_lr))
    latent_opt = torch.optim.Adam([latent_value], lr=float(latent_lr))
    methods = ["raw", "raw_from_decoded", "decoder_latent"]
    if bool(include_projected_raw_adam_tangent):
        methods.append("projected_raw_adam_tangent")
    path_lengths = {method: 0.0 for method in methods}
    last_thetas: dict[str, torch.Tensor | None] = {method: None for method in methods}
    last_steps: dict[str, torch.Tensor | None] = {method: None for method in methods}
    last_projected_info: dict[str, float] = {}

    curve_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    cg_rows: list[dict[str, Any]] = []

    for step in range(int(steps) + 1):
        batch_idx = _batch_indices(
            task_set,
            batch_size=batch_size,
            step=int(step),
            start_bank_position=start_bank_position,
        )
        with torch.no_grad():
            theta_raw = raw_value.detach()
            theta_decoded_raw = decoded_raw_value.detach()
            theta_latent = decode_weights(vae, normalizer, latent_value.detach().reshape(1, -1)).squeeze(0).detach()
            theta_projected = (
                decode_weights(vae, normalizer, projected_latent_value.detach().reshape(1, -1)).squeeze(0).detach()
                if bool(include_projected_raw_adam_tangent)
                else None
            )

        if int(step) in checkpoint_steps:
            checkpoint_thetas: list[tuple[str, torch.Tensor]] = [
                ("raw", theta_raw),
                ("raw_from_decoded", theta_decoded_raw),
                ("decoder_latent", theta_latent),
            ]
            if bool(include_projected_raw_adam_tangent) and theta_projected is not None:
                checkpoint_thetas.append(("projected_raw_adam_tangent", theta_projected))
            for method, theta in checkpoint_thetas:
                evals = _eval_path_state(theta=theta, task_set=task_set, spec=spec, tau=tau, penalty=float(cfg.finite_penalty))
                previous = last_thetas[method]
                last_step = last_steps[method]
                row = {
                    "label": label,
                    "run_name": run_name,
                    "method": method,
                    "source_weight_index": source_weight_index,
                    "start_bank_position": start_bank_position,
                    "task_name": task_name,
                    "tau": tau,
                    "step": int(step),
                    "raw_lr": float(raw_lr),
                    "latent_lr": float(latent_lr),
                    "batch_indices_sha256": _tensor_sha256(batch_idx),
                    "path_length": float(path_lengths[method]),
                    "path_length_rel_w0": _safe_ratio(float(path_lengths[method]), w0_norm),
                    "theta_rel_w0": _safe_ratio(_norm(theta - w0), w0_norm),
                    "theta_rel_dec0": _safe_ratio(_norm(theta - theta_dec0), w0_norm),
                    "last_step_norm": _norm(last_step) if last_step is not None else float("nan"),
                    "last_step_cos_with_prev_velocity": _cosine(theta - previous, last_step) if previous is not None and last_step is not None else float("nan"),
                    "last_raw_adam_proposed_step_norm": float(last_projected_info.get("projected_raw_adam_delta_norm", float("nan")))
                    if method == "projected_raw_adam_tangent"
                    else float("nan"),
                    "last_projected_linear_step_norm": float(last_projected_info.get("projected_raw_adam_linear_norm", float("nan")))
                    if method == "projected_raw_adam_tangent"
                    else float("nan"),
                    "last_projected_raw_adam_projection_residual_fraction": float(
                        last_projected_info.get("projected_raw_adam_projection_residual_fraction", float("nan"))
                    )
                    if method == "projected_raw_adam_tangent"
                    else float("nan"),
                    "decoded_start_reconstruction_rel_l2": _safe_ratio(_norm(theta_dec0 - w0), w0_norm),
                    **evals,
                }
                curve_rows.append(row)

            geom, cgs = _geometry_and_cg_rows(
                cfg=cfg,
                vae=vae,
                normalizer=normalizer,
                z=latent_value.detach(),
                theta=theta_latent,
                w0=w0,
                theta_dec0=theta_dec0,
                label=label,
                run_name=run_name,
                method="decoder_latent",
                start_row=start_row,
                step=int(step),
                raw_lr=float(raw_lr),
                latent_lr=float(latent_lr),
                task_set=task_set,
                spec=spec,
                tau=tau,
                batch_indices=batch_idx,
                lambda_abs=float(lambda_abs),
                tangent_damping_rel=float(tangent_damping_rel),
                probe_kind=str(probe_kind),
                probes=int(probes),
                seed=int(seed),
                cg_max_iters=int(cg_max_iters),
                cg_tol=float(cg_tol),
                cg_init=str(cg_init),
                jacobian_chunk_size=int(jacobian_chunk_size),
            )
            geometry_rows.append(geom)
            cg_rows.extend(cgs)
            if bool(include_projected_raw_adam_tangent) and theta_projected is not None:
                geom_projected, _ = _geometry_and_cg_rows(
                    cfg=cfg,
                    vae=vae,
                    normalizer=normalizer,
                    z=projected_latent_value.detach(),
                    theta=theta_projected,
                    w0=w0,
                    theta_dec0=theta_dec0,
                    label=label,
                    run_name=run_name,
                    method="projected_raw_adam_tangent",
                    start_row=start_row,
                    step=int(step),
                    raw_lr=float(raw_lr),
                    latent_lr=float(latent_lr),
                    task_set=task_set,
                    spec=spec,
                    tau=tau,
                    batch_indices=batch_idx,
                    lambda_abs=float(lambda_abs),
                    tangent_damping_rel=float(tangent_damping_rel),
                    probe_kind=str(probe_kind),
                    probes=0,
                    seed=int(seed),
                    cg_max_iters=int(cg_max_iters),
                    cg_tol=float(cg_tol),
                    cg_init=str(cg_init),
                    jacobian_chunk_size=int(jacobian_chunk_size),
                )
                geometry_rows.append(geom_projected)

        if int(step) == int(steps):
            break

        previous_raw = raw_value.detach().clone()
        previous_decoded_raw = decoded_raw_value.detach().clone()
        previous_latent_theta = theta_latent.detach().clone()
        previous_projected_theta = (
            theta_projected.detach().clone()
            if bool(include_projected_raw_adam_tangent) and theta_projected is not None
            else None
        )

        raw_opt.zero_grad(set_to_none=True)
        loss_raw, _ = _loss_acc(raw_value, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_idx)
        loss_raw.backward()
        raw_opt.step()

        decoded_raw_opt.zero_grad(set_to_none=True)
        loss_decoded_raw, _ = _loss_acc(
            decoded_raw_value,
            task_set=task_set,
            spec=spec,
            split="train",
            tau=tau,
            batch_indices=batch_idx,
        )
        loss_decoded_raw.backward()
        decoded_raw_opt.step()

        latent_opt.zero_grad(set_to_none=True)
        theta_latent_train = decode_weights(vae, normalizer, latent_value.reshape(1, -1)).squeeze(0)
        loss_latent, _ = _loss_acc(
            theta_latent_train,
            task_set=task_set,
            spec=spec,
            split="train",
            tau=tau,
            batch_indices=batch_idx,
        )
        loss_latent.backward()
        latent_opt.step()

        if bool(include_projected_raw_adam_tangent) and theta_projected is not None:
            projected_theta_req = theta_projected.detach().clone().requires_grad_(True)
            loss_projected, _ = _loss_acc(
                projected_theta_req,
                task_set=task_set,
                spec=spec,
                split="train",
                tau=tau,
                batch_indices=batch_idx,
            )
            projected_grad = torch.autograd.grad(loss_projected, projected_theta_req, retain_graph=False, create_graph=False)[0].detach()
            raw_adam_delta = _adam_delta_from_grad(grad=projected_grad, state=projected_adam_state, lr=float(raw_lr))
            projected_dz, last_projected_info = _project_x_delta_to_decoder_tangent(
                vae=vae,
                normalizer=normalizer,
                z=projected_latent_value,
                delta_theta=raw_adam_delta,
                damping_rel=float(tangent_damping_rel),
                jacobian_chunk_size=int(jacobian_chunk_size),
            )
            projected_latent_value = (projected_latent_value + projected_dz).detach()

        with torch.no_grad():
            new_raw = raw_value.detach()
            new_decoded_raw = decoded_raw_value.detach()
            new_latent_theta = decode_weights(vae, normalizer, latent_value.detach().reshape(1, -1)).squeeze(0).detach()
            new_projected_theta = (
                decode_weights(vae, normalizer, projected_latent_value.detach().reshape(1, -1)).squeeze(0).detach()
                if bool(include_projected_raw_adam_tangent)
                else None
            )
        updates = {
            "raw": new_raw - previous_raw,
            "raw_from_decoded": new_decoded_raw - previous_decoded_raw,
            "decoder_latent": new_latent_theta - previous_latent_theta,
        }
        if previous_projected_theta is not None and new_projected_theta is not None:
            updates["projected_raw_adam_tangent"] = new_projected_theta - previous_projected_theta
        for method, update in updates.items():
            path_lengths[method] += _norm(update)
            last_steps[method] = update.detach()
        last_thetas["raw"] = previous_raw
        last_thetas["raw_from_decoded"] = previous_decoded_raw
        last_thetas["decoder_latent"] = previous_latent_theta
        if previous_projected_theta is not None:
            last_thetas["projected_raw_adam_tangent"] = previous_projected_theta

    return curve_rows, geometry_rows, cg_rows


def _load_start_bank(path: Path, source_indices: list[int]) -> pd.DataFrame:
    bank = pd.read_csv(path)
    if "source_weight_index" not in bank.columns:
        raise ValueError(f"{path} has no source_weight_index column")
    if source_indices:
        order = {int(v): idx for idx, v in enumerate(source_indices)}
        selected = bank[bank["source_weight_index"].astype(int).isin(order)].copy()
        if len(selected) != len(order):
            got = set(selected["source_weight_index"].astype(int).tolist())
            missing = [int(v) for v in source_indices if int(v) not in got]
            raise ValueError(f"start bank missing source indices: {missing}")
        selected["_order"] = selected["source_weight_index"].astype(int).map(order)
        selected = selected.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    else:
        selected = bank.copy().reset_index(drop=True)
    return selected


def _paired_outputs(curves: pd.DataFrame, geometry: pd.DataFrame, cg: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    if len(labels) != 2:
        return
    control_label, a_label = labels
    curve_keys = ["method", "source_weight_index", "start_bank_position", "task_name", "tau", "step"]
    c = curves[curves["label"].astype(str) == control_label].copy()
    a = curves[curves["label"].astype(str) == a_label].copy()
    paired_curves = c.merge(a, on=curve_keys, suffixes=("_control", "_a"), validate="one_to_one")
    for col in ["train_loss", "test_loss", "theta_rel_w0", "theta_rel_dec0", "path_length_rel_w0"]:
        paired_curves[f"{col}_delta"] = paired_curves[f"{col}_a"] - paired_curves[f"{col}_control"]
    paired_curves.to_csv(out_dir / "trajectory_paired_curve_deltas.csv", index=False)

    geom_keys = ["source_weight_index", "start_bank_position", "task_name", "tau", "step"]
    if "method" in geometry.columns:
        geom_keys = ["method", *geom_keys]
    gc = geometry[geometry["label"].astype(str) == control_label].copy()
    ga = geometry[geometry["label"].astype(str) == a_label].copy()
    paired_geom = gc.merge(ga, on=geom_keys, suffixes=("_control", "_a"), validate="one_to_one")
    for col in [
        "trace_jtj_per_dim",
        "metric_effective_rank",
        "metric_condition",
        "projection_cos_with_neg_grad",
        "projection_norm_fraction",
        "normal_residual_fraction",
        "fresh_latent_adam_cos_with_neg_grad",
        "fresh_latent_adam_cos_with_natural_linear",
        "fresh_raw_adam_projection_residual_fraction",
    ]:
        paired_geom[f"{col}_delta"] = paired_geom[f"{col}_a"] - paired_geom[f"{col}_control"]
    paired_geom.to_csv(out_dir / "trajectory_paired_geometry_deltas.csv", index=False)

    cg_keys = ["source_weight_index", "start_bank_position", "task_name", "tau", "step", "probe_id"]
    if "method" in cg.columns:
        cg_keys = ["method", *cg_keys]
    cc = cg[cg["label"].astype(str) == control_label].copy()
    ca = cg[cg["label"].astype(str) == a_label].copy()
    paired_cg = cc.merge(ca, on=cg_keys, suffixes=("_control", "_a"), validate="one_to_one")
    for col in ["trace_jtj_per_dim", "c3_forward", "c3_inverse", "c3_total", "lambda_vtw_ratio"]:
        paired_cg[f"{col}_delta"] = paired_cg[f"{col}_a"] - paired_cg[f"{col}_control"]
        paired_cg[f"{col}_rel_delta"] = paired_cg[f"{col}_delta"] / paired_cg[f"{col}_control"].replace(0.0, np.nan)
    paired_cg["same_probe_sha256"] = paired_cg["probe_sha256_control"].astype(str) == paired_cg["probe_sha256_a"].astype(str)
    paired_cg["same_batch_indices_sha256"] = (
        paired_cg["batch_indices_sha256_control"].astype(str) == paired_cg["batch_indices_sha256_a"].astype(str)
    )
    paired_cg.to_csv(out_dir / "trajectory_paired_cg_deltas.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    cg_group_cols = ["step"]
    if "method" in paired_cg.columns:
        cg_group_cols = ["method", "step"]
    for group_key, group in paired_cg.groupby(cg_group_cols):
        if isinstance(group_key, tuple):
            method_name = str(group_key[0])
            step_value = int(group_key[1])
        else:
            method_name = "decoder_latent"
            step_value = int(group_key)
        for metric in ["c3_forward_rel_delta", "c3_inverse_rel_delta", "c3_total_rel_delta", "trace_jtj_per_dim_rel_delta"]:
            vals = group[metric].to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    "family": "fixed_cg",
                    "method": method_name,
                    "step": step_value,
                    "metric": metric,
                    "n": int(vals.size),
                    "mean": float(np.nanmean(vals)),
                    "median": float(np.nanmedian(vals)),
                    "better_count_delta_lt0": int(np.sum(vals < 0.0)),
                    "worse_count_delta_gt0": int(np.sum(vals > 0.0)),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
            )
    for method, method_group in paired_curves.groupby("method"):
        for step, group in method_group.groupby("step"):
            for metric in ["test_loss_delta", "train_loss_delta", "theta_rel_w0_delta", "path_length_rel_w0_delta"]:
                vals = group[metric].to_numpy(dtype=np.float64)
                summary_rows.append(
                    {
                        "family": f"{method}_curve",
                        "method": str(method),
                        "step": int(step),
                        "metric": metric,
                        "n": int(vals.size),
                        "mean": float(np.nanmean(vals)),
                        "median": float(np.nanmedian(vals)),
                        "better_count_delta_lt0": int(np.sum(vals < 0.0)),
                        "worse_count_delta_gt0": int(np.sum(vals > 0.0)),
                        "min": float(np.nanmin(vals)),
                        "max": float(np.nanmax(vals)),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "trajectory_step_summary.csv", index=False)

    validation = {
        "paired_curve_rows": int(len(paired_curves)),
        "paired_geometry_rows": int(len(paired_geom)),
        "paired_cg_rows": int(len(paired_cg)),
        "same_probe_sha256_all": bool(paired_cg["same_probe_sha256"].all()) if not paired_cg.empty else False,
        "same_batch_indices_sha256_all": bool(paired_cg["same_batch_indices_sha256"].all()) if not paired_cg.empty else False,
        "cg_all_finite": bool(
            np.isfinite(paired_cg[["c3_forward_a", "c3_inverse_a", "c3_total_a", "c3_forward_control", "c3_inverse_control", "c3_total_control"]].to_numpy()).all()
        )
        if not paired_cg.empty
        else False,
        "cg_residual_p90": float(
            pd.concat([paired_cg["cg_final_rel_residual_a"], paired_cg["cg_final_rel_residual_control"]]).quantile(0.90)
        )
        if not paired_cg.empty
        else float("nan"),
        "cg_residual_max": float(
            pd.concat([paired_cg["cg_final_rel_residual_a"], paired_cg["cg_final_rel_residual_control"]]).max()
        )
        if not paired_cg.empty
        else float("nan"),
    }
    (out_dir / "trajectory_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")
    if len(set(args.label)) != len(args.label):
        raise ValueError("--label values must be unique")
    run_dirs = [_run_dir(v) for v in args.run]
    labels = [str(v) for v in args.label]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_checkpoint_steps = args.checkpoint_step if args.checkpoint_step is not None else [1, 5, 25, 100, 300]
    checkpoint_steps = {int(v) for v in requested_checkpoint_steps}
    checkpoint_steps.add(0)
    checkpoint_steps.add(int(args.downstream_steps))
    source_indices = [int(v) for v in args.source_index]
    start_bank = _load_start_bank(Path(args.start_bank_csv).expanduser().resolve(), source_indices)
    start_bank.to_csv(out_dir / "trajectory_start_bank.csv", index=False)

    _log(
        "startup "
        f"device={args.device} labels={labels} starts={start_bank['source_weight_index'].astype(int).tolist()} "
        f"steps={args.downstream_steps} checkpoints={sorted(checkpoint_steps)} lambda_abs={float(args.lambda_abs):.8g} "
        f"probes={int(args.probes)} cg_max_iters={int(args.cg_max_iters)} output_dir={out_dir}"
    )
    base_cfg = _load_cfg(run_dirs[0], device=str(args.device), downstream_steps=int(args.downstream_steps))
    device = torch.device(base_cfg.device)
    dtype = torch_dtype(base_cfg)
    _log(f"resolved_config_hash={config_hash(base_cfg)} dtype={dtype} seed={base_cfg.seed}")
    _log("resolved_config=" + json.dumps(asdict(base_cfg), sort_keys=True, default=str))
    _log("stage=load_tasks")
    task_tensors = _load_task_tensors_for_pipeline(base_cfg, device=device, dtype=dtype)
    _log("loaded_tasks " + ", ".join(f"{k}:train={tuple(v.train_images.shape)} test={tuple(v.test_images.shape)}" for k, v in task_tensors.items()))
    weights_cpu, weight_records, weight_key, spec = _load_weight_pool(run_dirs[0])
    weights_device = weights_cpu.to(device=device, dtype=dtype)
    _log(f"stage=load_weight_pool weights_shape={tuple(weights_cpu.shape)} weight_key={weight_key} spec_dim={int(spec.dim)}")

    all_curves: list[pd.DataFrame] = []
    all_geometry: list[pd.DataFrame] = []
    all_cg: list[pd.DataFrame] = []
    selected_lrs: list[dict[str, Any]] = []
    for run_dir, label in zip(run_dirs, labels, strict=True):
        curve_path = out_dir / f"trajectory_curves_{label}.csv"
        geometry_path = out_dir / f"trajectory_geometry_{label}.csv"
        cg_path = out_dir / f"trajectory_cg_{label}.csv"
        if curve_path.is_file() and geometry_path.is_file() and cg_path.is_file() and not bool(args.force):
            _log(f"cache_hit label={label} curves={curve_path} geometry={geometry_path} cg={cg_path}")
            all_curves.append(pd.read_csv(curve_path))
            all_geometry.append(pd.read_csv(geometry_path))
            all_cg.append(pd.read_csv(cg_path))
            continue
        cfg = _load_cfg(run_dir, device=str(args.device), downstream_steps=int(args.downstream_steps))
        run_weights, _records, run_weight_key, _spec = _load_weight_pool(run_dir)
        if tuple(run_weights.shape) != tuple(weights_cpu.shape) or run_weight_key != weight_key:
            raise RuntimeError(f"weight pool mismatch for {run_dir}")
        vae, normalizer, vae_payload = _load_vae(run_dir, cfg, int(weights_cpu.shape[1]), device=device, dtype=dtype)
        raw_lr = _selected_lr(run_dir, "raw")
        latent_lr = _selected_lr(run_dir, "decoder_latent")
        selected_lrs.append({"label": label, "raw_lr": raw_lr, "decoder_latent_lr": latent_lr})
        _log(
            f"stage=trajectory label={label} run_dir={run_dir} raw_lr={raw_lr:g} latent_lr={latent_lr:g} "
            f"vae_cache_key={vae_payload.get('cache_key')}"
        )
        curve_rows: list[dict[str, Any]] = []
        geometry_rows: list[dict[str, Any]] = []
        cg_rows: list[dict[str, Any]] = []
        progress = make_progress(cfg, total=len(start_bank), desc=f"trajectory {label}")
        try:
            for pos, start_row in start_bank.iterrows():
                _log(
                    "start "
                    f"label={label} {pos + 1}/{len(start_bank)} source={int(start_row['source_weight_index'])} "
                    f"bank_pos={int(start_row['start_bank_position'])} task={start_row['task_name']} tau={float(start_row['tau']):.6g}"
                )
                curves, geometry, cg_rows_start = _run_start_paths(
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
                    steps=int(args.downstream_steps),
                    checkpoint_steps=checkpoint_steps,
                    lambda_abs=float(args.lambda_abs),
                    tangent_damping_rel=float(args.tangent_damping_rel),
                    probe_kind=str(args.probe_kind),
                    probes=int(args.probes),
                    seed=int(args.seed),
                    cg_max_iters=int(args.cg_max_iters),
                    cg_tol=float(args.cg_tol),
                    cg_init=str(args.cg_init),
                    jacobian_chunk_size=int(args.jacobian_chunk_size),
                    include_projected_raw_adam_tangent=bool(args.include_projected_raw_adam_tangent),
                )
                curve_rows.extend(curves)
                geometry_rows.extend(geometry)
                cg_rows.extend(cg_rows_start)
                latest = pd.DataFrame(cg_rows_start)
                if not latest.empty:
                    _log(
                        "start_done "
                        f"label={label} source={int(start_row['source_weight_index'])} "
                        f"median_forward={float(latest['c3_forward'].median()):.6g} "
                        f"median_residual={float(latest['cg_final_rel_residual'].median()):.3g}"
                    )
                progress.update(1)
        finally:
            progress.close()
        curves_frame = pd.DataFrame(curve_rows)
        geometry_frame = pd.DataFrame(geometry_rows)
        cg_frame = pd.DataFrame(cg_rows)
        curves_frame.to_csv(curve_path, index=False)
        geometry_frame.to_csv(geometry_path, index=False)
        cg_frame.to_csv(cg_path, index=False)
        all_curves.append(curves_frame)
        all_geometry.append(geometry_frame)
        all_cg.append(cg_frame)
        _log(
            f"label_done={label} curves={curve_path} rows={len(curves_frame)} "
            f"geometry={geometry_path} rows={len(geometry_frame)} cg={cg_path} rows={len(cg_frame)}"
        )

    curves_all = pd.concat(all_curves, ignore_index=True, sort=False)
    geometry_all = pd.concat(all_geometry, ignore_index=True, sort=False)
    cg_all = pd.concat(all_cg, ignore_index=True, sort=False)
    curves_all.to_csv(out_dir / "trajectory_curves.csv", index=False)
    geometry_all.to_csv(out_dir / "trajectory_geometry.csv", index=False)
    cg_all.to_csv(out_dir / "trajectory_cg.csv", index=False)
    pd.DataFrame(selected_lrs).to_csv(out_dir / "trajectory_selected_lrs.csv", index=False)
    _paired_outputs(curves_all, geometry_all, cg_all, labels, out_dir)
    manifest = {
        "runs": [str(v) for v in run_dirs],
        "labels": labels,
        "output_dir": str(out_dir),
        "start_bank_csv": str(Path(args.start_bank_csv).expanduser().resolve()),
        "source_indices": [int(v) for v in start_bank["source_weight_index"].astype(int).tolist()],
        "downstream_steps": int(args.downstream_steps),
        "checkpoint_steps": sorted(int(v) for v in checkpoint_steps),
        "lambda_abs": float(args.lambda_abs),
        "tangent_damping_rel": float(args.tangent_damping_rel),
        "probes": int(args.probes),
        "probe_kind": str(args.probe_kind),
        "cg_max_iters": int(args.cg_max_iters),
        "cg_tol": float(args.cg_tol),
        "cg_init": str(args.cg_init),
        "jacobian_chunk_size": int(args.jacobian_chunk_size),
        "include_projected_raw_adam_tangent": bool(args.include_projected_raw_adam_tangent),
        "projected_raw_adam_beta1": 0.9,
        "projected_raw_adam_beta2": 0.999,
        "projected_raw_adam_eps": 1.0e-8,
        "projected_raw_adam_line_search": "none",
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"done elapsed_sec={time.perf_counter() - started:.2f} output_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether Variant A's fixed-CG gains are realized along downstream trajectories.")
    parser.add_argument("--run", action="append", required=True, help="Run directory or run name. Repeat for labels.")
    parser.add_argument("--label", action="append", required=True, help="Label for each --run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-bank-csv", required=True)
    parser.add_argument("--source-index", action="append", type=int, default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--downstream-steps", type=int, default=300)
    parser.add_argument("--checkpoint-step", action="append", type=int, default=None)
    parser.add_argument("--lambda-abs", type=float, required=True)
    parser.add_argument("--tangent-damping-rel", type=float, default=1e-4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--probe-kind", default="rademacher", choices=("rademacher", "gaussian"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cg-max-iters", type=int, default=100)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--cg-init", default="lambda", choices=("zero", "lambda", "v_over_lambda", "null"))
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--include-projected-raw-adam-tangent", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
