from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class RealNVPConfig:
    dim: int
    num_layers: int = 8
    hidden_dim: int = 512
    network_depth: int = 2
    log_scale_clamp: float = 2.0
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class RQSplineFlowConfig:
    dim: int
    num_layers: int = 8
    hidden_dim: int = 64
    network_depth: int = 2
    num_bins: int = 8
    bound: float = 5.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    dropout: float = 0.0


def _make_alternating_mask(dim: int, parity: int, *, device: torch.device | None = None) -> torch.Tensor:
    indices = torch.arange(int(dim), device=device, dtype=torch.float32)
    return ((indices.long() + int(parity)) % 2 == 0).to(dtype=torch.float32)


def _make_layer_permutation(dim: int, layer_idx: int) -> torch.Tensor:
    if int(layer_idx) == 0:
        return torch.arange(int(dim), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(13_337 + int(layer_idx))
    return torch.randperm(int(dim), generator=generator, dtype=torch.long)


def _inverse_softplus(value: float) -> float:
    tensor = torch.as_tensor(float(value), dtype=torch.float64)
    return float((tensor.expm1()).log().item())


def _make_mlp(
    *,
    in_dim: int,
    hidden_dim: int,
    network_depth: int,
    out_dim: int,
    dropout: float,
) -> tuple[nn.Sequential, nn.Linear]:
    layers: list[nn.Module] = []
    current_dim = int(in_dim)
    depth = max(1, int(network_depth))
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.GELU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        current_dim = int(hidden_dim)
    out = nn.Linear(current_dim, int(out_dim))
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers), out


class AffineCouplingLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        network_depth: int,
        log_scale_clamp: float,
        mask_parity: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.log_scale_clamp = float(log_scale_clamp)
        self.register_buffer("mask", _make_alternating_mask(int(dim), int(mask_parity)), persistent=False)

        self.net, self.final_linear = _make_mlp(
            in_dim=int(dim),
            hidden_dim=int(hidden_dim),
            network_depth=int(network_depth),
            out_dim=2 * int(dim),
            dropout=float(dropout),
        )

    def _shift_and_log_scale(self, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(fixed).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.log_scale_clamp
        inv_mask = 1.0 - self.mask.to(device=fixed.device, dtype=fixed.dtype)
        return shift * inv_mask, log_scale * inv_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or int(x.shape[1]) != self.dim:
            raise ValueError(f"x must be [B,{self.dim}], got {tuple(x.shape)}")
        mask = self.mask.to(device=x.device, dtype=x.dtype)
        fixed = x * mask
        shift, log_scale = self._shift_and_log_scale(fixed)
        inv_mask = 1.0 - mask
        y = fixed + inv_mask * (x * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.ndim != 2 or int(y.shape[1]) != self.dim:
            raise ValueError(f"y must be [B,{self.dim}], got {tuple(y.shape)}")
        mask = self.mask.to(device=y.device, dtype=y.dtype)
        fixed = y * mask
        shift, log_scale = self._shift_and_log_scale(fixed)
        inv_mask = 1.0 - mask
        x = fixed + inv_mask * ((y - shift) * torch.exp(-log_scale))
        log_det = -log_scale.sum(dim=-1)
        return x, log_det


class RealNVPFlow(nn.Module):
    """Identity-initialized RealNVP-style flow for flat latent vectors."""

    def __init__(self, cfg: RealNVPConfig) -> None:
        super().__init__()
        if int(cfg.dim) <= 0:
            raise ValueError(f"flow dim must be positive, got {cfg.dim}")
        if int(cfg.num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {cfg.num_layers}")
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                AffineCouplingLayer(
                    dim=int(cfg.dim),
                    hidden_dim=int(cfg.hidden_dim),
                    network_depth=int(cfg.network_depth),
                    log_scale_clamp=float(cfg.log_scale_clamp),
                    mask_parity=idx % 2,
                    dropout=float(cfg.dropout),
                )
                for idx in range(int(cfg.num_layers))
            ]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z.new_zeros(int(z.shape[0]))
        out = z
        for layer in self.layers:
            out, layer_log_det = layer(out)
            log_det = log_det + layer_log_det
        return out, log_det

    def inverse(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z_flat.new_zeros(int(z_flat.shape[0]))
        out = z_flat
        for layer in reversed(self.layers):
            out, layer_log_det = layer.inverse(out)
            log_det = log_det + layer_log_det
        return out, log_det


def _gather_last(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)


def _normalize_spline_bins(
    logits: torch.Tensor,
    *,
    min_value: float,
    total: float,
) -> torch.Tensor:
    num_bins = int(logits.shape[-1])
    min_total = float(min_value) * num_bins
    if min_total >= float(total):
        raise ValueError(f"num_bins * min_value must be smaller than total: {min_total} >= {total}")
    return float(min_value) + (float(total) - min_total) * torch.softmax(logits, dim=-1)


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    *,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    bound: float,
    inverse: bool,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be [B,T], got {tuple(inputs.shape)}")
    if int(inputs.shape[-1]) == 0:
        return inputs, inputs.new_zeros(int(inputs.shape[0]))

    num_bins = int(widths.shape[-1])
    if heights.shape != widths.shape:
        raise ValueError(f"heights shape must match widths shape, got {tuple(heights.shape)} vs {tuple(widths.shape)}")
    if derivatives.shape != (*widths.shape[:-1], num_bins + 1):
        raise ValueError(f"derivatives must end with K+1, got {tuple(derivatives.shape)}")

    total = 2.0 * float(bound)
    zeros = widths.new_zeros(*widths.shape[:-1], 1)
    cumwidths_raw = torch.cat([zeros, torch.cumsum(widths, dim=-1)], dim=-1) - float(bound)
    cumheights_raw = torch.cat([zeros, torch.cumsum(heights, dim=-1)], dim=-1) - float(bound)
    cumwidths = torch.cat(
        [
            cumwidths_raw[..., :1].new_full(cumwidths_raw[..., :1].shape, -float(bound)),
            cumwidths_raw[..., 1:-1],
            cumwidths_raw[..., -1:].new_full(cumwidths_raw[..., -1:].shape, float(bound)),
        ],
        dim=-1,
    )
    cumheights = torch.cat(
        [
            cumheights_raw[..., :1].new_full(cumheights_raw[..., :1].shape, -float(bound)),
            cumheights_raw[..., 1:-1],
            cumheights_raw[..., -1:].new_full(cumheights_raw[..., -1:].shape, float(bound)),
        ],
        dim=-1,
    )

    inside = (inputs >= -float(bound)) & (inputs <= float(bound))
    calc_inputs = inputs.clamp(min=-float(bound), max=float(bound))
    bin_locations = cumheights if inverse else cumwidths
    bin_idx = torch.sum(calc_inputs.unsqueeze(-1) >= bin_locations[..., 1:-1], dim=-1)
    bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)

    input_cumwidths = _gather_last(cumwidths, bin_idx)
    input_bin_widths = _gather_last(widths, bin_idx)
    input_cumheights = _gather_last(cumheights, bin_idx)
    input_bin_heights = _gather_last(heights, bin_idx)
    delta = input_bin_heights / input_bin_widths
    derivative_left = _gather_last(derivatives, bin_idx)
    derivative_right = _gather_last(derivatives, bin_idx + 1)

    if inverse:
        y_minus_cumheight = calc_inputs - input_cumheights
        common = derivative_left + derivative_right - 2.0 * delta
        a = y_minus_cumheight * common + input_bin_heights * (delta - derivative_left)
        b = input_bin_heights * derivative_left - y_minus_cumheight * common
        c = -delta * y_minus_cumheight
        discriminant = (b.square() - 4.0 * a * c).clamp_min(float(eps))
        sqrt_discriminant = torch.sqrt(discriminant)
        denominator = -b - sqrt_discriminant
        denominator = torch.where(
            denominator.abs() < float(eps),
            denominator.sign().masked_fill(denominator == 0, -1.0) * float(eps),
            denominator,
        )
        theta = ((2.0 * c) / denominator).clamp(0.0, 1.0)
        calc_outputs = input_cumwidths + theta * input_bin_widths
    else:
        theta = ((calc_inputs - input_cumwidths) / input_bin_widths).clamp(0.0, 1.0)
        theta_one_minus_theta = theta * (1.0 - theta)
        numerator = input_bin_heights * (
            delta * theta.square() + derivative_left * theta_one_minus_theta
        )
        denominator = delta + (derivative_left + derivative_right - 2.0 * delta) * theta_one_minus_theta
        calc_outputs = input_cumheights + numerator / denominator.clamp_min(float(eps))

    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = delta + (derivative_left + derivative_right - 2.0 * delta) * theta_one_minus_theta
    derivative_numerator = delta.square() * (
        derivative_right * theta.square()
        + 2.0 * delta * theta_one_minus_theta
        + derivative_left * (1.0 - theta).square()
    )
    derivative_denominator = denominator.square()
    forward_logabsdet = torch.log(derivative_numerator.clamp_min(float(eps))) - torch.log(
        derivative_denominator.clamp_min(float(eps))
    )
    calc_logabsdet = -forward_logabsdet if inverse else forward_logabsdet

    outputs = torch.where(inside, calc_outputs, inputs)
    logabsdet = torch.where(inside, calc_logabsdet, torch.zeros_like(calc_logabsdet))
    # The normalization above should make the interval exactly [-B, B], but keep this
    # assertion local to shape/scale bugs rather than floating-point endpoint drift.
    if total <= 0.0:
        raise ValueError(f"spline bound must be positive, got {bound}")
    return outputs, logabsdet.sum(dim=-1)


class RQSplineCouplingLayer(nn.Module):
    """Near-identity coupling layer with monotone rational-quadratic splines."""

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        network_depth: int,
        num_bins: int,
        bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
        mask_parity: int,
        layer_idx: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(dim) < 2:
            raise ValueError(f"RQ-spline coupling flow requires dim >= 2, got {dim}")
        if int(num_bins) < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        if float(bound) <= 0.0:
            raise ValueError(f"bound must be positive, got {bound}")
        total = 2.0 * float(bound)
        if int(num_bins) * float(min_bin_width) >= total:
            raise ValueError("num_bins * min_bin_width must be smaller than 2 * bound")
        if int(num_bins) * float(min_bin_height) >= total:
            raise ValueError("num_bins * min_bin_height must be smaller than 2 * bound")
        if float(min_derivative) <= 0.0:
            raise ValueError(f"min_derivative must be positive, got {min_derivative}")

        self.dim = int(dim)
        self.num_bins = int(num_bins)
        self.bound = float(bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        permutation = _make_layer_permutation(int(dim), int(layer_idx))
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(int(dim), dtype=torch.long)
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer("inverse_permutation", inverse_permutation, persistent=False)

        mask = _make_alternating_mask(int(dim), int(mask_parity)).to(dtype=torch.bool)
        transform_mask = ~mask
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("fixed_indices", torch.nonzero(mask, as_tuple=False).flatten().to(dtype=torch.long), persistent=False)
        self.register_buffer(
            "transform_indices",
            torch.nonzero(transform_mask, as_tuple=False).flatten().to(dtype=torch.long),
            persistent=False,
        )
        self.transform_count = int(transform_mask.sum().item())
        if self.transform_count <= 0:
            raise ValueError("coupling layer must transform at least one coordinate")

        output_dim = self.transform_count * (3 * self.num_bins + 1)
        self.net, self.final_linear = _make_mlp(
            in_dim=int(dim),
            hidden_dim=int(hidden_dim),
            network_depth=int(network_depth),
            out_dim=output_dim,
            dropout=float(dropout),
        )
        derivative_bias = _inverse_softplus(1.0 - self.min_derivative)
        self.register_buffer("derivative_bias", torch.as_tensor(derivative_bias, dtype=torch.float32), persistent=False)

    def _params(self, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(fixed)
        raw = raw.reshape(int(fixed.shape[0]), self.transform_count, 3 * self.num_bins + 1)
        width_logits = raw[..., : self.num_bins]
        height_logits = raw[..., self.num_bins : 2 * self.num_bins]
        derivative_raw = raw[..., 2 * self.num_bins :]
        widths = _normalize_spline_bins(
            width_logits,
            min_value=self.min_bin_width,
            total=2.0 * self.bound,
        )
        heights = _normalize_spline_bins(
            height_logits,
            min_value=self.min_bin_height,
            total=2.0 * self.bound,
        )
        derivative_bias = self.derivative_bias.to(device=fixed.device, dtype=fixed.dtype)
        derivatives = self.min_derivative + F.softplus(derivative_raw + derivative_bias)
        return widths, heights, derivatives

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or int(x.shape[1]) != self.dim:
            raise ValueError(f"x must be [B,{self.dim}], got {tuple(x.shape)}")
        x_perm = x.index_select(dim=-1, index=self.permutation.to(device=x.device))
        mask = self.mask.to(device=x.device)
        fixed = x_perm * mask.to(dtype=x.dtype)
        widths, heights, derivatives = self._params(fixed)
        transform_indices = self.transform_indices.to(device=x.device)
        transformed = x_perm.index_select(dim=-1, index=transform_indices)
        transformed_out, log_det = _rational_quadratic_spline(
            transformed,
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            bound=self.bound,
            inverse=False,
        )
        scatter_indices = transform_indices.unsqueeze(0).expand(int(x.shape[0]), -1)
        y_perm = x_perm.scatter(dim=-1, index=scatter_indices, src=transformed_out)
        y = y_perm.index_select(dim=-1, index=self.inverse_permutation.to(device=x.device))
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.ndim != 2 or int(y.shape[1]) != self.dim:
            raise ValueError(f"y must be [B,{self.dim}], got {tuple(y.shape)}")
        y_perm = y.index_select(dim=-1, index=self.permutation.to(device=y.device))
        mask = self.mask.to(device=y.device)
        fixed = y_perm * mask.to(dtype=y.dtype)
        widths, heights, derivatives = self._params(fixed)
        transform_indices = self.transform_indices.to(device=y.device)
        transformed = y_perm.index_select(dim=-1, index=transform_indices)
        transformed_out, log_det = _rational_quadratic_spline(
            transformed,
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            bound=self.bound,
            inverse=True,
        )
        scatter_indices = transform_indices.unsqueeze(0).expand(int(y.shape[0]), -1)
        x_perm = y_perm.scatter(dim=-1, index=scatter_indices, src=transformed_out)
        x = x_perm.index_select(dim=-1, index=self.inverse_permutation.to(device=y.device))
        return x, log_det


class RQSplineFlow(nn.Module):
    """Primary near-identity rational-quadratic spline coupling flow."""

    def __init__(self, cfg: RQSplineFlowConfig) -> None:
        super().__init__()
        if int(cfg.dim) <= 1:
            raise ValueError(f"RQ-spline flow dim must be at least 2, got {cfg.dim}")
        if int(cfg.num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {cfg.num_layers}")
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                RQSplineCouplingLayer(
                    dim=int(cfg.dim),
                    hidden_dim=int(cfg.hidden_dim),
                    network_depth=int(cfg.network_depth),
                    num_bins=int(cfg.num_bins),
                    bound=float(cfg.bound),
                    min_bin_width=float(cfg.min_bin_width),
                    min_bin_height=float(cfg.min_bin_height),
                    min_derivative=float(cfg.min_derivative),
                    mask_parity=idx % 2,
                    layer_idx=idx,
                    dropout=float(cfg.dropout),
                )
                for idx in range(int(cfg.num_layers))
            ]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z.new_zeros(int(z.shape[0]))
        out = z
        for layer in self.layers:
            out, layer_log_det = layer(out)
            log_det = log_det + layer_log_det
        return out, log_det

    def inverse(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z_flat.new_zeros(int(z_flat.shape[0]))
        out = z_flat
        for layer in reversed(self.layers):
            out, layer_log_det = layer.inverse(out)
            log_det = log_det + layer_log_det
        return out, log_det


def _jacobian_single(fn, sample: torch.Tensor) -> torch.Tensor:
    sample_req = sample.detach().clone().requires_grad_(True)
    output = fn(sample_req).reshape(-1)
    rows: list[torch.Tensor] = []
    for idx in range(int(output.numel())):
        grad = torch.autograd.grad(output[idx], sample_req, retain_graph=True, create_graph=False)[0]
        rows.append(grad.reshape(-1))
    return torch.stack(rows, dim=0)


def _flow_jacobians_for_sanity(flow: nn.Module, samples: torch.Tensor) -> torch.Tensor:
    def flow_forward_single(sample: torch.Tensor) -> torch.Tensor:
        return flow(sample.unsqueeze(0))[0].squeeze(0)

    func = getattr(torch, "func", None)
    if func is not None and hasattr(func, "jacrev") and hasattr(func, "vmap"):
        try:
            return func.vmap(func.jacrev(flow_forward_single))(samples.detach())
        except RuntimeError:
            pass
    return torch.stack([_jacobian_single(flow_forward_single, samples[idx]) for idx in range(int(samples.shape[0]))], dim=0)


def flow_architecture_sanity(
    flow: nn.Module,
    samples: torch.Tensor,
    *,
    compute_condition: bool = True,
    condition_samples: int = 8,
    eps: float = 1e-12,
) -> dict[str, float]:
    if samples.ndim != 2:
        raise ValueError(f"samples must be [N,D], got {tuple(samples.shape)}")
    if int(samples.shape[0]) <= 0:
        raise ValueError("samples must contain at least one row")
    flow_was_training = flow.training
    flow.eval()
    try:
        with torch.no_grad():
            u, log_det = flow(samples)
            reconstructed, inverse_log_det = flow.inverse(u)
            roundtrip = (reconstructed - samples).abs()
            displacement = (u - samples).abs()
            metrics = {
                "roundtrip_max_abs_error": float(roundtrip.max().detach().cpu().item()),
                "median_abs_displacement": float(displacement.median().detach().cpu().item()),
                "median_abs_logdet": float(log_det.abs().median().detach().cpu().item()),
                "median_abs_logdet_roundtrip_sum": float((log_det + inverse_log_det).abs().median().detach().cpu().item()),
            }
        if compute_condition:
            jac_samples = samples[: min(int(condition_samples), int(samples.shape[0]))].detach()
            jac = _flow_jacobians_for_sanity(flow, jac_samples)
            singular_values = torch.linalg.svdvals(jac.float()).detach()
            cond = singular_values.max(dim=-1).values / singular_values.min(dim=-1).values.clamp_min(float(eps))
            metrics["median_condition"] = float(cond.median().detach().cpu().item())
        return metrics
    finally:
        flow.train(flow_was_training)
