from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from post_train_research.big_vae_latent_flattening.flow import RealNVPConfig, RealNVPFlow

from .config import ExperimentConfig
from .probe_geometry import TensorFn, isometry_objective_from_jacobians, probe_jacobians_for_flow


LOGGER = logging.getLogger("flow_preconditioning")


def make_flow(dim: int, cfg: ExperimentConfig) -> RealNVPFlow:
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


def make_random_flow(dim: int, cfg: ExperimentConfig, *, device: torch.device, dtype: torch.dtype, seed: int) -> RealNVPFlow:
    flow = make_flow(int(dim), cfg).to(device=device, dtype=dtype)
    std = float(cfg.random_flow_near_identity_noise_std)
    if std <= 0.0:
        return flow
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for layer in flow.layers:
            final_linear = None
            for module in layer.net.modules():
                if isinstance(module, torch.nn.Linear) and int(module.out_features) == 2 * int(dim):
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
    flow: RealNVPFlow
    history: list[dict[str, float]]
    elapsed_s: float


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
    torch.manual_seed(int(seed))
    flow = make_flow(dim, cfg).to(device=device, dtype=dtype)
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
