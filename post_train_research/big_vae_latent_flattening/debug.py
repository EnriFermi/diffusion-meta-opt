from __future__ import annotations

import json
import logging
import math
from typing import Any, Iterator

import torch

from post_train_research.big_vae_latent_flattening.config import RunConfig
from post_train_research.big_vae_latent_flattening.flow import RealNVPFlow


def _debug_number(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0.0 else "-inf"
    return number


def _tensor_debug_stats(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"name": str(name), "present": False}
    with torch.no_grad():
        detached = tensor.detach()
        values = detached.abs() if detached.is_complex() else detached
        values = values.to(dtype=torch.float32)
        flat_values = values.reshape(-1)
        finite_mask = torch.isfinite(flat_values)
        nan_count = int(torch.isnan(flat_values).sum().item())
        posinf_count = int(torch.logical_and(torch.isinf(flat_values), flat_values > 0).sum().item())
        neginf_count = int(torch.logical_and(torch.isinf(flat_values), flat_values < 0).sum().item())
        finite_count = int(finite_mask.sum().item())
        numel = int(flat_values.numel())
        stats: dict[str, Any] = {
            "name": str(name),
            "present": True,
            "shape": [int(dim) for dim in detached.shape],
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": numel,
            "finite_count": finite_count,
            "nan_count": nan_count,
            "posinf_count": posinf_count,
            "neginf_count": neginf_count,
            "all_finite": finite_count == numel,
        }
        if finite_count > 0:
            finite_values = flat_values[finite_mask]
            stats.update(
                {
                    "finite_min": _debug_number(finite_values.min().item()),
                    "finite_max": _debug_number(finite_values.max().item()),
                    "finite_mean": _debug_number(finite_values.mean().item()),
                    "finite_std": _debug_number(finite_values.std(unbiased=False).item()),
                    "finite_abs_max": _debug_number(finite_values.abs().max().item()),
                    "finite_l2": _debug_number(torch.linalg.vector_norm(finite_values).item()),
                }
            )
        nonfinite_indices = torch.nonzero(~finite_mask, as_tuple=False).flatten()[:8]
        if int(nonfinite_indices.numel()) > 0:
            stats["first_nonfinite_flat_indices"] = [int(idx) for idx in nonfinite_indices.cpu().tolist()]
        return stats


def _compact_tensor_debug_stats(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    stats = _tensor_debug_stats(name, tensor)
    return {
        key: stats.get(key)
        for key in (
            "name",
            "present",
            "shape",
            "numel",
            "finite_count",
            "nan_count",
            "posinf_count",
            "neginf_count",
            "all_finite",
            "finite_min",
            "finite_max",
            "finite_mean",
            "finite_abs_max",
            "finite_l2",
            "first_nonfinite_flat_indices",
        )
        if key in stats
    }


def _sort_debug_number(value: Any) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return -1.0


def _top_named_tensor_debug_rows(
    named_tensors: Iterator[tuple[str, torch.Tensor | None]],
    *,
    topk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        stats = _compact_tensor_debug_stats(name, tensor)
        bad_count = (
            int(stats.get("nan_count", 0))
            + int(stats.get("posinf_count", 0))
            + int(stats.get("neginf_count", 0))
        )
        stats["bad_count"] = bad_count
        rows.append(stats)
    rows.sort(
        key=lambda row: (
            int(row.get("bad_count", 0)),
            _sort_debug_number(row.get("finite_abs_max")),
            int(row.get("numel", 0)),
        ),
        reverse=True,
    )
    return rows[: max(1, int(topk))]


def _flow_layer_debug_rows(flow: RealNVPFlow, z: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = z.detach()
    with torch.no_grad():
        for layer_idx, layer in enumerate(flow.layers):
            row: dict[str, Any] = {
                "layer": int(layer_idx),
                "input": _compact_tensor_debug_stats("input", current),
            }
            try:
                if hasattr(layer, "mask") and callable(getattr(layer, "_shift_and_log_scale", None)):
                    mask = layer.mask.to(device=current.device, dtype=current.dtype)
                    fixed = current * mask
                    shift, log_scale = layer._shift_and_log_scale(fixed)
                    row["shift"] = _compact_tensor_debug_stats("shift", shift)
                    row["log_scale"] = _compact_tensor_debug_stats("log_scale", log_scale)
                current, layer_log_det = layer(current)
                row["output"] = _compact_tensor_debug_stats("output", current)
                row["log_det"] = _compact_tensor_debug_stats("log_det", layer_log_det)
            except Exception as exc:
                row["error"] = repr(exc)
                rows.append(row)
                break
            rows.append(row)
    return rows


def _json_for_log(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str, allow_nan=False)


def should_emit_nonfinite_debug(consecutive_nonfinite_steps: int) -> bool:
    return int(consecutive_nonfinite_steps) <= 3 or int(consecutive_nonfinite_steps) % 10 == 0


def log_nonfinite_debug(
    *,
    logger: logging.Logger,
    step: int,
    reason: str,
    consecutive_nonfinite_steps: int,
    cfg: RunConfig,
    flow: RealNVPFlow,
    batch: Any,
    state: Any,
    z_prime: torch.Tensor,
    forward_log_det: torch.Tensor,
    distortion: Any,
    z_norm_loss: torch.Tensor,
    flow_cycle_mse: torch.Tensor,
    loss: torch.Tensor,
    include_grads: bool,
) -> None:
    topk = max(1, int(cfg.train.nonfinite_debug_topk))
    tensors = {
        "batch.W": _tensor_debug_stats("batch.W", batch.W),
        "batch.X": _tensor_debug_stats("batch.X", batch.X),
        "batch.x_mask": _tensor_debug_stats("batch.x_mask", batch.x_mask),
        "batch.d_in_mask": _tensor_debug_stats("batch.d_in_mask", batch.d_in_mask),
        "batch.d_out_mask": _tensor_debug_stats("batch.d_out_mask", batch.d_out_mask),
        "state.z": _tensor_debug_stats("state.z", state.z),
        "state.logvar": _tensor_debug_stats("state.logvar", state.logvar),
        "state.dist_patch_by_patch": _tensor_debug_stats("state.dist_patch_by_patch", state.dist_patch_by_patch),
        "state.patch_mask": _tensor_debug_stats("state.patch_mask", state.patch_mask),
        "state.d_in_mask": _tensor_debug_stats("state.d_in_mask", state.d_in_mask),
        "state.d_out_mask": _tensor_debug_stats("state.d_out_mask", state.d_out_mask),
        "z_prime": _tensor_debug_stats("z_prime", z_prime),
        "forward_log_det": _tensor_debug_stats("forward_log_det", forward_log_det),
        "distortion.loss": _tensor_debug_stats("distortion.loss", distortion.loss),
        "distortion.ratio": _tensor_debug_stats("distortion.ratio", distortion.ratio),
        "distortion.zero_baseline": _tensor_debug_stats("distortion.zero_baseline", distortion.zero_baseline),
        "distortion.trace_g": _tensor_debug_stats("distortion.trace_g", distortion.trace_g),
        "distortion.trace_g2": _tensor_debug_stats("distortion.trace_g2", distortion.trace_g2),
        "z_norm_loss": _tensor_debug_stats("z_norm_loss", z_norm_loss),
        "flow_cycle_mse": _tensor_debug_stats("flow_cycle_mse", flow_cycle_mse),
        "loss": _tensor_debug_stats("loss", loss),
    }
    logger.warning(
        "Non-finite latent flattening debug tensors: step=%s reason=%s consecutive=%s payload=%s",
        step,
        reason,
        consecutive_nonfinite_steps,
        _json_for_log(tensors),
    )
    logger.warning(
        "Non-finite latent flattening debug flow_layers: step=%s reason=%s payload=%s",
        step,
        reason,
        _json_for_log(_flow_layer_debug_rows(flow, state.z)),
    )
    logger.warning(
        "Non-finite latent flattening debug flow_params_top: step=%s reason=%s payload=%s",
        step,
        reason,
        _json_for_log(_top_named_tensor_debug_rows(flow.named_parameters(), topk=topk)),
    )
    if include_grads:
        grad_rows = _top_named_tensor_debug_rows(
            ((name, param.grad) for name, param in flow.named_parameters()),
            topk=topk,
        )
        logger.warning(
            "Non-finite latent flattening debug flow_grads_top: step=%s reason=%s payload=%s",
            step,
            reason,
            _json_for_log(grad_rows),
        )
