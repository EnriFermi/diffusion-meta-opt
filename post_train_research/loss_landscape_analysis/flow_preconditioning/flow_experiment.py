from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import torch

from post_train_research.big_vae_latent_flattening.flow import (
    RQSplineFlow,
    RQSplineFlowConfig,
    RealNVPConfig,
    RealNVPFlow,
    flow_architecture_sanity,
)

from .config import ExperimentConfig
from .probe_geometry import TensorFn, isometry_objective_from_jacobians, probe_jacobians_for_flow


LOGGER = logging.getLogger("flow_preconditioning")
_FLOW_INIT_LOCK = threading.Lock()


def flow_architecture_name(cfg: ExperimentConfig) -> str:
    return str(getattr(cfg, "flow_architecture", "rq_spline")).strip().lower().replace("-", "_")


def make_flow(dim: int, cfg: ExperimentConfig) -> torch.nn.Module:
    architecture = flow_architecture_name(cfg)
    if architecture in {"rq_spline", "rqspline", "rational_quadratic_spline", "spline"}:
        return RQSplineFlow(
            RQSplineFlowConfig(
                dim=int(dim),
                num_layers=int(cfg.flow_num_layers),
                hidden_dim=int(cfg.flow_hidden_dim),
                network_depth=int(cfg.flow_network_depth),
                num_bins=int(cfg.flow_spline_bins),
                bound=float(cfg.flow_spline_bound),
                min_bin_width=float(cfg.flow_spline_min_bin_width),
                min_bin_height=float(cfg.flow_spline_min_bin_height),
                min_derivative=float(cfg.flow_spline_min_derivative),
                dropout=float(cfg.flow_dropout),
            )
        )
    if architecture in {"realnvp", "affine", "affine_coupling"}:
        return RealNVPFlow(
            RealNVPConfig(
                dim=int(dim),
                num_layers=int(cfg.flow_num_layers),
                hidden_dim=int(cfg.flow_hidden_dim),
                network_depth=int(cfg.flow_network_depth),
                log_scale_clamp=float(cfg.flow_log_scale_clamp),
                dropout=float(cfg.flow_dropout),
            )
        )
    raise ValueError(f"unknown flow architecture {cfg.flow_architecture!r}")


def make_random_flow(dim: int, cfg: ExperimentConfig, *, device: torch.device, dtype: torch.dtype, seed: int) -> torch.nn.Module:
    flow = make_flow(int(dim), cfg).to(device=device, dtype=dtype)
    std = float(cfg.random_flow_near_identity_noise_std)
    if std <= 0.0:
        return flow
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for layer in getattr(flow, "layers", []):
            final_linear = getattr(layer, "final_linear", None)
            if not isinstance(final_linear, torch.nn.Linear):
                final_linear = None
                for module in layer.modules():
                    if isinstance(module, torch.nn.Linear):
                        final_linear = module
            if final_linear is None:
                continue
            weight = torch.randn(final_linear.weight.shape, generator=generator, dtype=dtype) * std
            bias = torch.randn(final_linear.bias.shape, generator=generator, dtype=dtype) * std
            final_linear.weight.copy_(weight.to(device=device))
            final_linear.bias.copy_(bias.to(device=device))
    return flow


def freeze_flow(flow: torch.nn.Module) -> torch.nn.Module:
    flow.eval()
    for param in flow.parameters():
        param.requires_grad_(False)
    return flow


@dataclass(slots=True)
class FlowTrainingResult:
    flow: torch.nn.Module
    history: list[dict[str, float]]
    elapsed_s: float


def assert_flow_architecture_sanity(
    flow: torch.nn.Module,
    samples: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    log_label: str = "",
) -> dict[str, float]:
    sample_count = min(max(1, int(cfg.flow_sanity_samples)), int(samples.shape[0]))
    metrics = flow_architecture_sanity(
        flow,
        samples[:sample_count],
        compute_condition=bool(cfg.flow_sanity_compute_condition),
        condition_samples=max(1, int(cfg.flow_sanity_condition_samples)),
    )
    label = f"{log_label} " if log_label else ""
    LOGGER.info(
        "%sflow_architecture_sanity architecture=%s samples=%d roundtrip_max=%.6g median_abs_displacement=%.6g "
        "median_abs_logdet=%.6g median_abs_logdet_roundtrip_sum=%.6g median_cond=%s",
        label,
        flow_architecture_name(cfg),
        sample_count,
        metrics["roundtrip_max_abs_error"],
        metrics["median_abs_displacement"],
        metrics["median_abs_logdet"],
        metrics["median_abs_logdet_roundtrip_sum"],
        f"{metrics['median_condition']:.6g}" if "median_condition" in metrics else "skipped",
    )
    failures: list[str] = []
    if metrics["roundtrip_max_abs_error"] > float(cfg.flow_sanity_max_roundtrip_error):
        failures.append(
            f"roundtrip_max_abs_error={metrics['roundtrip_max_abs_error']:.6g} "
            f"> {float(cfg.flow_sanity_max_roundtrip_error):.6g}"
        )
    if metrics["median_abs_displacement"] > float(cfg.flow_sanity_max_median_abs_displacement):
        failures.append(
            f"median_abs_displacement={metrics['median_abs_displacement']:.6g} "
            f"> {float(cfg.flow_sanity_max_median_abs_displacement):.6g}"
        )
    if metrics["median_abs_logdet"] > float(cfg.flow_sanity_max_median_abs_logdet):
        failures.append(
            f"median_abs_logdet={metrics['median_abs_logdet']:.6g} "
            f"> {float(cfg.flow_sanity_max_median_abs_logdet):.6g}"
        )
    if "median_condition" in metrics and metrics["median_condition"] > float(cfg.flow_sanity_max_median_condition):
        failures.append(
            f"median_condition={metrics['median_condition']:.6g} "
            f"> {float(cfg.flow_sanity_max_median_condition):.6g}"
        )
    if failures:
        raise RuntimeError(f"flow architecture sanity failed for {flow_architecture_name(cfg)}: {', '.join(failures)}")
    return metrics


def train_flow(
    *,
    probe: TensorFn,
    theta_samples: torch.Tensor,
    cfg: ExperimentConfig,
    steps: int,
    seed: int,
    log_label: str = "",
) -> FlowTrainingResult:
    if theta_samples.ndim != 2:
        raise ValueError(f"theta_samples must be [N,D], got {tuple(theta_samples.shape)}")
    dim = int(theta_samples.shape[1])
    device = theta_samples.device
    dtype = theta_samples.dtype
    with _FLOW_INIT_LOCK:
        torch.manual_seed(int(seed))
        flow = make_flow(dim, cfg).to(device=device, dtype=dtype)
    assert_flow_architecture_sanity(flow, theta_samples, cfg, log_label=log_label)
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(cfg.flow_lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    history: list[dict[str, float]] = []
    started = time.time()
    sample_count = int(theta_samples.shape[0])
    batch_size = min(max(1, int(cfg.flow_batch_size)), sample_count)
    label = f"{log_label} " if log_label else ""
    LOGGER.info(
        "%sflow_train start dim=%d samples=%d batch_size=%d steps=%d lr=%g",
        label,
        dim,
        sample_count,
        batch_size,
        int(steps),
        float(cfg.flow_lr),
    )
    for step in range(1, int(steps) + 1):
        indices = torch.randint(0, sample_count, (batch_size,), generator=generator, device="cpu")
        batch = theta_samples.index_select(0, indices.to(device=theta_samples.device))
        optimizer.zero_grad(set_to_none=True)
        jacobians, _u = probe_jacobians_for_flow(probe, flow, batch, create_graph=True)
        loss = isometry_objective_from_jacobians(jacobians, dim=dim)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite flow isometry loss at step {step}: {float(loss.detach().cpu().item())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), float(cfg.flow_grad_clip_norm))
        optimizer.step()
        if step == 1 or step % int(cfg.flow_log_every) == 0 or step == int(steps):
            with torch.no_grad():
                loss_value = float(loss.detach().cpu().item())
                grad_value = float(torch.as_tensor(grad_norm).detach().cpu().item())
                elapsed = float(time.time() - started)
                history.append(
                    {
                        "step": float(step),
                        "loss": loss_value,
                        "grad_norm": grad_value,
                        "elapsed_s": elapsed,
                    }
                )
                LOGGER.info(
                    "%sflow_train step=%d/%d loss=%.6g grad=%.6g elapsed_s=%.1f",
                    label,
                    step,
                    int(steps),
                    loss_value,
                    grad_value,
                    elapsed,
                )
    elapsed_total = float(time.time() - started)
    LOGGER.info("%sflow_train done elapsed_s=%.1f", label, elapsed_total)
    return FlowTrainingResult(flow=flow, history=history, elapsed_s=elapsed_total)
