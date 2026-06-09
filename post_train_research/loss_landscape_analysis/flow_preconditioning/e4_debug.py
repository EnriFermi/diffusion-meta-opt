from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ExperimentConfig, config_to_dict, torch_dtype
from .flow_experiment import assert_flow_architecture_sanity, make_flow
from .optimization import Curve, LossFn
from .probe_geometry import (
    ResidualThetaProbe,
    TensorFn,
    exact_jacobians,
    fit_probe_scales,
    isometry_objective_from_jacobians,
    metric_tensors_from_jacobians,
    probe_jacobians_for_flow,
    probe_jacobians_for_theta,
)
from .toy_mlp import (
    MLP_DIM,
    MLPRegressionProblem,
    collect_mlp_flow_pool,
    collect_mlp_heldout_geometry_pool,
)


@dataclass(frozen=True, slots=True)
class E4DebugConfig:
    run_label: str = "e4_geometry_debug"
    artifact_root: str = "artifacts/loss_landscape_analysis/flow_preconditioning/e4_debug"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    seed: int = 0
    rho: float = 1e-2
    flow_steps: int = 300
    eval_every: int = 25
    flow_batch_size: int = 16
    flow_lr: float = 1e-3
    flow_grad_clip_norm: float = 10.0
    flow_architecture: str = "rq_spline"
    flow_num_layers: int = 8
    flow_hidden_dim: int = 64
    flow_network_depth: int = 2
    flow_log_scale_clamp: float = 1.5
    flow_spline_bins: int = 8
    flow_spline_bound: float = 5.0
    flow_spline_min_bin_width: float = 1e-3
    flow_spline_min_bin_height: float = 1e-3
    flow_spline_min_derivative: float = 1e-3
    flow_dropout: float = 0.0
    flow_random_samples: int = 128
    flow_trajectory_count: int = 8
    flow_trajectory_steps: int = 8
    flow_trajectory_lr: float = 1e-2
    heldout_geometry_samples: int = 64
    train_eval_samples: int = 32
    heldout_eval_samples: int = 32
    init_std: float = 0.5
    train_points: int = 128
    probe_points: int = 64
    test_points: int = 512
    train_shift: float = 0.0
    probe_shift: float = 0.37
    test_shift: float = 0.73


@dataclass(slots=True)
class E4DebugState:
    cfg: ExperimentConfig
    debug_cfg: E4DebugConfig
    output_dir: Path
    problem: MLPRegressionProblem
    probe: ResidualThetaProbe
    flow_pool: torch.Tensor
    heldout_pool: torch.Tensor
    train_eval_pool: torch.Tensor
    heldout_eval_pool: torch.Tensor


@dataclass(slots=True)
class E4DebugResult:
    flow: torch.nn.Module
    history: pd.DataFrame
    final_geometry: pd.DataFrame
    output_dir: Path


def experiment_config_from_debug(debug_cfg: E4DebugConfig) -> ExperimentConfig:
    return ExperimentConfig(
        run_label=str(debug_cfg.run_label),
        artifact_root=str(debug_cfg.artifact_root),
        cache_first=False,
        force_rerun=True,
        device=str(debug_cfg.device),
        dtype=str(debug_cfg.dtype),
        experiments=("E4",),
        seeds=(int(debug_cfg.seed),),
        e4_rho_values=(float(debug_cfg.rho),),
        e4_main_rho=float(debug_cfg.rho),
        init_std=float(debug_cfg.init_std),
        train_points=int(debug_cfg.train_points),
        probe_points=int(debug_cfg.probe_points),
        test_points=int(debug_cfg.test_points),
        train_shift=float(debug_cfg.train_shift),
        probe_shift=float(debug_cfg.probe_shift),
        test_shift=float(debug_cfg.test_shift),
        flow_random_samples=int(debug_cfg.flow_random_samples),
        flow_trajectory_count=int(debug_cfg.flow_trajectory_count),
        flow_trajectory_steps=int(debug_cfg.flow_trajectory_steps),
        flow_trajectory_lr=float(debug_cfg.flow_trajectory_lr),
        heldout_geometry_samples=int(debug_cfg.heldout_geometry_samples),
        flow_steps=int(debug_cfg.flow_steps),
        e4_flow_steps=int(debug_cfg.flow_steps),
        flow_batch_size=int(debug_cfg.flow_batch_size),
        flow_lr=float(debug_cfg.flow_lr),
        flow_grad_clip_norm=float(debug_cfg.flow_grad_clip_norm),
        flow_log_every=max(1, int(debug_cfg.eval_every)),
        flow_architecture=str(debug_cfg.flow_architecture),
        flow_num_layers=int(debug_cfg.flow_num_layers),
        flow_hidden_dim=int(debug_cfg.flow_hidden_dim),
        flow_network_depth=int(debug_cfg.flow_network_depth),
        flow_log_scale_clamp=float(debug_cfg.flow_log_scale_clamp),
        flow_spline_bins=int(debug_cfg.flow_spline_bins),
        flow_spline_bound=float(debug_cfg.flow_spline_bound),
        flow_spline_min_bin_width=float(debug_cfg.flow_spline_min_bin_width),
        flow_spline_min_bin_height=float(debug_cfg.flow_spline_min_bin_height),
        flow_spline_min_derivative=float(debug_cfg.flow_spline_min_derivative),
        flow_dropout=float(debug_cfg.flow_dropout),
        save_figures=False,
        progress_log_enabled=False,
    )


def build_e4_debug_state(debug_cfg: E4DebugConfig) -> E4DebugState:
    cfg = experiment_config_from_debug(debug_cfg)
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    output_dir = Path(debug_cfg.artifact_root).expanduser().resolve() / str(debug_cfg.run_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = MLPRegressionProblem(cfg=cfg, seed=int(debug_cfg.seed), device=device, dtype=dtype)
    flow_pool = collect_mlp_flow_pool(problem, cfg, seed=int(debug_cfg.seed))
    heldout_pool = collect_mlp_heldout_geometry_pool(problem, cfg, seed=int(debug_cfg.seed))
    scales = fit_probe_scales(problem.probe_residual, flow_pool)
    probe = ResidualThetaProbe(problem.probe_residual, scales=scales, rho=float(debug_cfg.rho), eps=float(cfg.aulc_eps))
    train_eval_pool = flow_pool[: min(int(debug_cfg.train_eval_samples), int(flow_pool.shape[0]))].detach()
    heldout_eval_pool = heldout_pool[: min(int(debug_cfg.heldout_eval_samples), int(heldout_pool.shape[0]))].detach()

    (output_dir / "debug_config.json").write_text(
        json.dumps(
            {
                "debug_config": asdict(debug_cfg),
                "experiment_config": config_to_dict(cfg),
                "flow_pool_shape": list(flow_pool.shape),
                "heldout_pool_shape": list(heldout_pool.shape),
                "probe_scales": asdict(scales),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return E4DebugState(
        cfg=cfg,
        debug_cfg=debug_cfg,
        output_dir=output_dir,
        problem=problem,
        probe=probe,
        flow_pool=flow_pool,
        heldout_pool=heldout_pool,
        train_eval_pool=train_eval_pool,
        heldout_eval_pool=heldout_eval_pool,
    )


def _geometry_metrics(jacobians: torch.Tensor, *, prefix: str, eps: float = 1e-12) -> dict[str, float]:
    metric, trace_g, trace_g2 = metric_tensors_from_jacobians(jacobians.detach())
    eig = torch.linalg.eigvalsh(metric.float()).detach()
    eig_min = eig.min(dim=-1).values
    eig_max = eig.max(dim=-1).values
    cond = eig_max / eig_min.clamp_min(float(eps))
    dim = int(jacobians.shape[-1])
    r_value = isometry_objective_from_jacobians(jacobians.detach(), dim=dim, eps=eps)
    return {
        f"{prefix}_R": float(r_value.detach().cpu().item()),
        f"{prefix}_trace_mean": float(trace_g.mean().detach().cpu().item()),
        f"{prefix}_trace_cv": float((trace_g.float().std(unbiased=False) / trace_g.float().mean().abs().clamp_min(float(eps))).cpu().item()),
        f"{prefix}_trace2_mean": float(trace_g2.mean().detach().cpu().item()),
        f"{prefix}_eig_min_median": float(eig_min.median().detach().cpu().item()),
        f"{prefix}_eig_max_median": float(eig_max.median().detach().cpu().item()),
        f"{prefix}_cond_median": float(cond.median().detach().cpu().item()),
        f"{prefix}_cond_p90": float(torch.quantile(cond.float(), 0.90).detach().cpu().item()),
    }


def _flow_coordinate_metrics(flow: torch.nn.Module, samples: torch.Tensor, *, prefix: str, eps: float = 1e-12) -> dict[str, float]:
    def flow_forward(theta: torch.Tensor) -> torch.Tensor:
        return flow(theta.unsqueeze(0))[0].squeeze(0)

    jac = exact_jacobians(flow_forward, samples, create_graph=False).detach()
    mapped, log_det = flow(samples)
    svals = torch.linalg.svdvals(jac.float()).detach()
    cond = svals.max(dim=-1).values / svals.min(dim=-1).values.clamp_min(float(eps))
    displacement = (mapped - samples).detach().float().norm(dim=-1)
    return {
        f"{prefix}_flow_cond_median": float(cond.median().cpu().item()),
        f"{prefix}_flow_cond_p90": float(torch.quantile(cond, 0.90).cpu().item()),
        f"{prefix}_flow_displacement_median": float(displacement.median().cpu().item()),
        f"{prefix}_flow_displacement_p90": float(torch.quantile(displacement, 0.90).cpu().item()),
        f"{prefix}_flow_logdet_mean": float(log_det.detach().float().mean().cpu().item()),
        f"{prefix}_flow_logdet_std": float(log_det.detach().float().std(unbiased=False).cpu().item()),
    }


def evaluate_original_geometry(probe: TensorFn, samples: torch.Tensor, *, prefix: str) -> dict[str, float]:
    jac = probe_jacobians_for_theta(probe, samples, create_graph=False)
    return _geometry_metrics(jac, prefix=prefix)


def evaluate_flow_geometry(probe: TensorFn, flow: torch.nn.Module, samples: torch.Tensor, *, prefix: str) -> dict[str, float]:
    jac, _ = probe_jacobians_for_flow(probe, flow, samples, create_graph=False)
    metrics = _geometry_metrics(jac, prefix=prefix)
    metrics.update(_flow_coordinate_metrics(flow, samples, prefix=prefix))
    return metrics


def evaluate_debug_snapshot(state: E4DebugState, flow: torch.nn.Module, *, step: int, batch_loss: float, elapsed_s: float) -> dict[str, float]:
    flow_was_training = flow.training
    flow.eval()
    try:
        row: dict[str, Any] = {
            "step": int(step),
            "batch_loss": float(batch_loss),
            "elapsed_s": float(elapsed_s),
        }
        row.update(evaluate_original_geometry(state.probe, state.train_eval_pool, prefix="train_original"))
        row.update(evaluate_flow_geometry(state.probe, flow, state.train_eval_pool, prefix="train_flow"))
        row.update(evaluate_original_geometry(state.probe, state.heldout_eval_pool, prefix="heldout_original"))
        row.update(evaluate_flow_geometry(state.probe, flow, state.heldout_eval_pool, prefix="heldout_flow"))
        row["train_R_delta"] = row["train_flow_R"] - row["train_original_R"]
        row["heldout_R_delta"] = row["heldout_flow_R"] - row["heldout_original_R"]
        row["train_R_ratio"] = row["train_flow_R"] / max(row["train_original_R"], 1e-12)
        row["heldout_R_ratio"] = row["heldout_flow_R"] / max(row["heldout_original_R"], 1e-12)
        return row
    finally:
        flow.train(flow_was_training)


def train_e4_debug_flow(state: E4DebugState) -> E4DebugResult:
    cfg = state.cfg
    debug_cfg = state.debug_cfg
    torch.manual_seed(int(debug_cfg.seed) + 50_000)
    flow = make_flow(MLP_DIM, cfg).to(device=state.flow_pool.device, dtype=state.flow_pool.dtype)
    assert_flow_architecture_sanity(flow, state.flow_pool, cfg, log_label="[E4 debug]")
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(debug_cfg.flow_lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(debug_cfg.seed) + 50_017)
    sample_count = int(state.flow_pool.shape[0])
    batch_size = min(max(1, int(debug_cfg.flow_batch_size)), sample_count)
    eval_every = max(1, int(debug_cfg.eval_every))
    rows: list[dict[str, float]] = []
    started = time.time()

    rows.append(evaluate_debug_snapshot(state, flow, step=0, batch_loss=float("nan"), elapsed_s=0.0))
    flow.train()
    for step in range(1, int(debug_cfg.flow_steps) + 1):
        indices = torch.randint(0, sample_count, (batch_size,), generator=generator, device="cpu")
        batch = state.flow_pool.index_select(0, indices.to(device=state.flow_pool.device))
        optimizer.zero_grad(set_to_none=True)
        jacobians, _ = probe_jacobians_for_flow(state.probe, flow, batch, create_graph=True)
        loss = isometry_objective_from_jacobians(jacobians, dim=MLP_DIM)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite E4 debug flow loss at step {step}: {float(loss.detach().cpu().item())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), float(debug_cfg.flow_grad_clip_norm))
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == int(debug_cfg.flow_steps):
            row = evaluate_debug_snapshot(
                state,
                flow,
                step=step,
                batch_loss=float(loss.detach().cpu().item()),
                elapsed_s=time.time() - started,
            )
            row["grad_norm"] = float(torch.as_tensor(grad_norm).detach().cpu().item())
            rows.append(row)
            pd.DataFrame(rows).to_csv(state.output_dir / "history.csv", index=False)

    history = pd.DataFrame(rows)
    final_rows = [
        {"split": "train_eval", "coordinate": "original", **evaluate_original_geometry(state.probe, state.train_eval_pool, prefix="metric")},
        {"split": "train_eval", "coordinate": "flow", **evaluate_flow_geometry(state.probe, flow, state.train_eval_pool, prefix="metric")},
        {"split": "heldout_eval", "coordinate": "original", **evaluate_original_geometry(state.probe, state.heldout_eval_pool, prefix="metric")},
        {"split": "heldout_eval", "coordinate": "flow", **evaluate_flow_geometry(state.probe, flow, state.heldout_eval_pool, prefix="metric")},
    ]
    final_geometry = pd.DataFrame(final_rows)
    history.to_csv(state.output_dir / "history.csv", index=False)
    final_geometry.to_csv(state.output_dir / "final_geometry.csv", index=False)
    torch.save(flow.state_dict(), state.output_dir / "flow_state.pt")
    return E4DebugResult(flow=flow, history=history, final_geometry=final_geometry, output_dir=state.output_dir)


def run_probe_metric_curve(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    probe: TensorFn,
    theta0: torch.Tensor,
    lr: float,
    steps: int,
    damping: float,
    store_path: bool = False,
) -> Curve:
    theta = theta0.detach().clone().requires_grad_(True)
    dim = int(theta.numel())
    train_losses_device = torch.empty(int(steps) + 1, device=theta.device, dtype=torch.float64)
    test_losses_device = torch.empty(int(steps) + 1, device=theta.device, dtype=torch.float64)
    path = torch.empty(int(steps) + 1, dim, device=theta.device, dtype=torch.float64) if store_path else None
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)

    for step in range(0, int(steps) + 1):
        loss = train_loss_fn(theta)
        train_losses_device[step] = loss.detach().to(dtype=torch.float64)
        with torch.no_grad():
            test_losses_device[step] = test_loss_fn(theta).detach().to(dtype=torch.float64)
            if path is not None:
                path[step] = theta.detach().to(dtype=torch.float64)
        if step == int(steps):
            break

        grad = torch.autograd.grad(loss, theta)[0].detach()
        jacobian = probe_jacobians_for_theta(probe, theta.detach().reshape(1, -1), create_graph=False).squeeze(0)
        metric = jacobian.transpose(0, 1) @ jacobian
        metric = metric + float(damping) * eye
        direction = torch.linalg.solve(metric, grad.reshape(-1, 1)).reshape(-1)
        theta = (theta.detach() - float(lr) * direction.detach()).requires_grad_(True)

    return Curve(
        train_loss=train_losses_device.detach().cpu().numpy(),
        test_loss=test_losses_device.detach().cpu().numpy(),
        final_theta=theta.detach().cpu().numpy(),
        path=path.detach().cpu().numpy() if path is not None else None,
    )


def run_probe_metric_curves(
    *,
    train_loss_fn: LossFn,
    test_loss_fn: LossFn,
    probe: TensorFn,
    theta0_batch: torch.Tensor,
    lr: float,
    steps: int,
    damping: float,
) -> list[Curve]:
    return [
        run_probe_metric_curve(
            train_loss_fn=train_loss_fn,
            test_loss_fn=test_loss_fn,
            probe=probe,
            theta0=theta0_batch[idx],
            lr=float(lr),
            steps=int(steps),
            damping=float(damping),
        )
        for idx in range(int(theta0_batch.shape[0]))
    ]


def _safe_cosine(lhs: torch.Tensor, rhs: torch.Tensor, *, eps: float = 1e-12) -> float:
    lhs_flat = lhs.detach().reshape(-1).float()
    rhs_flat = rhs.detach().reshape(-1).float()
    denom = lhs_flat.norm() * rhs_flat.norm()
    if float(denom.cpu().item()) <= float(eps):
        return float("nan")
    return float((lhs_flat @ rhs_flat / denom.clamp_min(float(eps))).cpu().item())


def _preconditioner_vectors(
    *,
    train_loss_fn: LossFn,
    probe: TensorFn,
    flow: torch.nn.Module,
    theta: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_value = theta.detach().reshape(-1)
    dim = int(theta_value.numel())
    theta_req = theta_value.clone().requires_grad_(True)
    loss = train_loss_fn(theta_req)
    grad = torch.autograd.grad(loss, theta_req)[0].detach()

    probe_jacobian = probe_jacobians_for_theta(probe, theta_value.reshape(1, -1), create_graph=False).squeeze(0)
    probe_metric = probe_jacobian.transpose(0, 1) @ probe_jacobian
    eye = torch.eye(dim, device=theta_value.device, dtype=theta_value.dtype)
    p_metric = torch.linalg.solve(probe_metric + float(damping) * eye, grad.reshape(-1, 1)).reshape(-1)

    flow_was_training = flow.training
    flow.eval()
    try:
        with torch.no_grad():
            u_value = flow(theta_value.reshape(1, -1))[0].reshape(-1).detach()

        def inverse_at_u(u: torch.Tensor) -> torch.Tensor:
            return flow.inverse(u.unsqueeze(0))[0].squeeze(0)

        inverse_jacobian = exact_jacobians(inverse_at_u, u_value.reshape(1, -1), create_graph=False).squeeze(0)
        p_nf = inverse_jacobian @ (inverse_jacobian.transpose(0, 1) @ grad)
    finally:
        flow.train(flow_was_training)
    return loss.detach(), grad, p_metric, p_nf


def preconditioner_alignment_row(
    *,
    train_loss_fn: LossFn,
    probe: TensorFn,
    flow: torch.nn.Module,
    theta: torch.Tensor,
    damping: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    loss, grad, p_metric, p_nf = _preconditioner_vectors(
        train_loss_fn=train_loss_fn,
        probe=probe,
        flow=flow,
        theta=theta,
        damping=float(damping),
    )
    grad_norm = grad.float().norm()
    metric_norm = p_metric.float().norm()
    nf_norm = p_nf.float().norm()
    return {
        "loss": float(loss.cpu().item()),
        "grad_norm": float(grad_norm.cpu().item()),
        "p_metric_norm": float(metric_norm.cpu().item()),
        "p_nf_norm": float(nf_norm.cpu().item()),
        "norm_ratio_metric_to_nf": float((metric_norm / nf_norm.clamp_min(float(eps))).cpu().item()),
        "cos_p_nf_p_metric": _safe_cosine(p_nf, p_metric, eps=eps),
        "cos_g_p_metric": _safe_cosine(grad, p_metric, eps=eps),
        "cos_g_p_nf": _safe_cosine(grad, p_nf, eps=eps),
    }


def preconditioner_alignment_rows(
    *,
    train_loss_fn: LossFn,
    probe: TensorFn,
    flow: torch.nn.Module,
    theta_points: torch.Tensor,
    damping: float,
    source: str,
    start_index: int,
    steps: list[int] | tuple[int, ...],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    if theta_points.ndim != 2:
        raise ValueError(f"theta_points must be [N,D], got {tuple(theta_points.shape)}")
    if len(steps) != int(theta_points.shape[0]):
        raise ValueError(f"steps length must match theta_points rows: {len(steps)} vs {int(theta_points.shape[0])}")
    for row_idx in range(int(theta_points.shape[0])):
        row = preconditioner_alignment_row(
            train_loss_fn=train_loss_fn,
            probe=probe,
            flow=flow,
            theta=theta_points[row_idx],
            damping=float(damping),
        )
        rows.append(
            {
                "source": str(source),
                "start_index": int(start_index),
                "step": int(steps[row_idx]),
                **row,
            }
        )
    return rows


def trajectory_step_alignment_rows(
    *,
    train_loss_fn: LossFn,
    probe: TensorFn,
    flow: torch.nn.Module,
    theta_path: torch.Tensor,
    damping: float,
    source: str,
    start_index: int,
    steps: list[int] | tuple[int, ...],
    eps: float = 1e-12,
) -> list[dict[str, float | int | str]]:
    if theta_path.ndim != 2:
        raise ValueError(f"theta_path must be [N,D], got {tuple(theta_path.shape)}")
    if len(steps) != int(theta_path.shape[0]):
        raise ValueError(f"steps length must match theta_path rows: {len(steps)} vs {int(theta_path.shape[0])}")
    rows: list[dict[str, float | int | str]] = []
    for row_idx in range(int(theta_path.shape[0]) - 1):
        loss, grad, p_metric, p_nf = _preconditioner_vectors(
            train_loss_fn=train_loss_fn,
            probe=probe,
            flow=flow,
            theta=theta_path[row_idx],
            damping=float(damping),
        )
        # Optimizers apply theta_next = theta - update. The descent-space step is therefore theta - theta_next.
        descent_step = theta_path[row_idx].detach().reshape(-1) - theta_path[row_idx + 1].detach().reshape(-1)
        step_norm = descent_step.float().norm()
        metric_norm = p_metric.float().norm()
        nf_norm = p_nf.float().norm()
        rows.append(
            {
                "source": str(source),
                "start_index": int(start_index),
                "step": int(steps[row_idx]),
                "next_step": int(steps[row_idx + 1]),
                "loss": float(loss.cpu().item()),
                "step_norm": float(step_norm.cpu().item()),
                "grad_norm": float(grad.float().norm().cpu().item()),
                "p_metric_norm": float(metric_norm.cpu().item()),
                "p_nf_norm": float(nf_norm.cpu().item()),
                "cos_step_p_metric": _safe_cosine(descent_step, p_metric, eps=eps),
                "cos_step_p_nf": _safe_cosine(descent_step, p_nf, eps=eps),
                "step_norm_over_p_metric_norm": float((step_norm / metric_norm.clamp_min(float(eps))).cpu().item()),
                "step_norm_over_p_nf_norm": float((step_norm / nf_norm.clamp_min(float(eps))).cpu().item()),
            }
        )
    return rows
