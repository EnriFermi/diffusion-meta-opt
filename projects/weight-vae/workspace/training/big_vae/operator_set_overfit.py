from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import torch

from big_vae.datasets.operator_bank import (
    CommittedBundleSampler,
    OperatorBankBundleDataset,
    OperatorBankTrainingDataset,
)
from training.big_vae.two_operator_overfit import _loss_metrics, _nrmse, _raw_model, _stitch


def _weighted_cosine_position_oracle(
    target_matrices: torch.Tensor,
    *,
    patch_size: int,
    gamma: float,
) -> torch.Tensor:
    """Best shared patch direction for the exact weighted structural L_dir."""
    if target_matrices.ndim != 3:
        raise ValueError("target_matrices must be [operators,d_in,d_out]")
    operator_count, matrix_rows, matrix_cols = target_matrices.shape
    if matrix_rows % int(patch_size) != 0:
        raise ValueError("position oracle requires complete input patches")
    patches = target_matrices.float().transpose(1, 2).reshape(
        operator_count,
        matrix_cols,
        matrix_rows // int(patch_size),
        int(patch_size),
    )
    radii = patches.norm(dim=-1)
    units = patches / radii.unsqueeze(-1).clamp_min(1.0e-12)
    structural_weights = radii.clamp_min(1.0e-12).pow(float(gamma))
    structural_weights = structural_weights / structural_weights.sum(
        dim=2, keepdim=True
    ).clamp_min(1.0e-12)
    oracle_units = (structural_weights.unsqueeze(-1) * units).sum(dim=0)
    oracle_units = oracle_units / oracle_units.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    # Radius does not affect L_dir. Mean radius makes scale diagnostics descriptive,
    # but this artifact is named and interpreted only as the direction oracle.
    oracle_radii = radii.mean(dim=0)
    return (oracle_units * oracle_radii.unsqueeze(-1)).reshape(
        matrix_cols, matrix_rows
    ).transpose(0, 1)


def v11_complement_normalized_mse(
    target: torch.Tensor,
    prediction: torch.Tensor,
    latent_mu: torch.Tensor,
    *,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    num_latents: int = 32,
    d_latent: int = 384,
    protected_dim: int = 320,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Scale-sensitive V11 error normalized by target complement energy.

    V11 reconstructs an exact orthonormal row-space floor. Therefore
    ``prediction-target`` is exactly the learned complement error, while
    ``||target||²-||zc||²`` is the target complement energy.
    """

    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("V11 complement loss requires matching [B,d_in,d_out] tensors")
    batch, d_in, d_out = target.shape
    if tuple(d_in_mask.shape) != (batch, d_in) or tuple(d_out_mask.shape) != (
        batch,
        d_out,
    ):
        raise ValueError("V11 complement loss mask shapes do not match target")
    if tuple(latent_mu.shape) != (batch, num_latents * d_latent):
        raise ValueError("V11 complement loss latent shape violates the serialized contract")
    valid = (
        d_in_mask.to(device=target.device, dtype=torch.bool).unsqueeze(-1)
        & d_out_mask.to(device=target.device, dtype=torch.bool).unsqueeze(1)
    )
    with torch.autocast(device_type=target.device.type, enabled=False):
        target_f = target.float() * valid
        prediction_f = prediction.float() * valid
        protected = latent_mu.detach().view(batch, num_latents, d_latent)[
            ..., :protected_dim
        ].float()
        target_energy = target_f.square().sum(dim=(-1, -2))
        protected_energy = protected.square().sum(dim=(-1, -2))
        complement_energy = (target_energy - protected_energy).clamp_min(float(eps))
        error_energy = (prediction_f - target_f).square().sum(dim=(-1, -2))
        return (error_energy / complement_energy).mean()


def validate_v9a_step0_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Fail closed before update if V9-A is not carrier-dominant and W-live."""
    ledger = {
        "mixed_adaptive_to_carrier_rms": float(
            metrics.get("v9a_mixed_adaptive_to_carrier_rms", float("inf"))
        ),
        "total_z_shuffle_output_relative_delta": float(
            metrics.get("z_shuffle_output_relative_delta", 0.0)
        ),
        "total_fixed_x_w_shuffle_output_relative_delta": float(
            metrics.get("fixed_x_w_shuffle_output_relative_delta", 0.0)
        ),
        "isolated_adaptive_output_relative_delta": float(
            metrics.get("v9a_full_vs_carrier_only_output_relative_delta", 0.0)
        ),
        "isolated_adaptive_shuffle_output_relative_delta": float(
            metrics.get("v9a_adaptive_shuffle_output_relative_delta", 0.0)
        ),
        "zero_w_latent_max_abs": float(
            metrics.get("v9a_zero_w_latent_max_abs", float("inf"))
        ),
    }
    ratio = ledger["mixed_adaptive_to_carrier_rms"]
    if not 0.2 <= ratio <= 0.5:
        raise RuntimeError(
            "V9-A step-0 mixed adaptive/carrier energy is outside [0.2,0.5]: "
            f"ratio={ratio:.6g}"
        )
    if min(
        ledger["isolated_adaptive_output_relative_delta"],
        ledger["isolated_adaptive_shuffle_output_relative_delta"],
    ) < 1.0e-3:
        raise RuntimeError(
            "V9-A step-0 isolated adaptive bridge/head path is too weak: "
            f"full_vs_carrier={ledger['isolated_adaptive_output_relative_delta']:.6g} "
            f"adaptive_shuffle={ledger['isolated_adaptive_shuffle_output_relative_delta']:.6g}"
        )
    if min(
        ledger["total_z_shuffle_output_relative_delta"],
        ledger["total_fixed_x_w_shuffle_output_relative_delta"],
    ) < 1.0e-3:
        raise RuntimeError(
            "V9-A step-0 total carrier+adaptive identity path is too weak: "
            f"z={ledger['total_z_shuffle_output_relative_delta']:.6g} "
            f"fixed_x_w={ledger['total_fixed_x_w_shuffle_output_relative_delta']:.6g}"
        )
    if ledger["zero_w_latent_max_abs"] != 0.0:
        raise RuntimeError(
            "V9-A full encoder is not exactly zero-preserving for W=0: "
            f"max_abs={ledger['zero_w_latent_max_abs']:.6g}"
        )
    return ledger


def validate_v9a_step1_gradients(
    model: torch.nn.Module,
    *,
    adam_eps: float,
) -> dict[str, Any]:
    """Require every V9-A block's exact B18 content/router path above Adam eps."""
    raw_model = _raw_model(model)
    refiner = getattr(raw_model, "hybrid_content_readout_v9", None)
    if refiner is None or not bool(getattr(refiner, "carrier_mean_content", False)):
        raise RuntimeError("V9-A step-1 gradient gate requires the carrier-mean refiner")
    if not 0.0 < float(adam_eps) < 1.0:
        raise ValueError("Adam eps must be finite and positive")
    groups: dict[str, float] = {}
    for block_index, block in enumerate(refiner.blocks):
        modules = {
            "query": block.query_proj,
            "weight_key": block.weight_key_proj,
            "weight_norm_key": block.weight_norm_key_proj,
            "context_key": block.context_key_proj,
            "raw_correction": block.out_proj,
            "ffn_in": block.local_ffn[0],
            "ffn_out": block.local_ffn[2],
        }
        for group_name, module in modules.items():
            grad = module.weight.grad
            key = f"block_{block_index:02d}.{group_name}"
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise RuntimeError(f"V9-A step-1 gradient missing/nonfinite: {key}")
            rms = float(grad.float().square().mean().sqrt().item())
            groups[key] = rms
            if rms <= float(adam_eps):
                raise RuntimeError(
                    f"V9-A step-1 gradient is at/below Adam eps: {key} "
                    f"grad_rms={rms:.6g} adam_eps={float(adam_eps):.6g}"
                )
    return {
        "schema": "weightclip_ae_v9a_exact_b18_step1_gradients_v1",
        "adam_eps": float(adam_eps),
        "minimum_group_grad_rms": min(groups.values()),
        "groups": groups,
    }


def validate_v10_geometry_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Fail closed on invariants that must hold at every V10 evaluation."""

    names = (
        "v10_basis_orthogonality_max_abs",
        "v10_basis_min_valid_column_coverage",
        "v10_zc_analysis_max_abs",
        "v10_residual_rowspace_max_abs",
        "v10_floor_residual_inner_product_max_abs",
        "v10_floor_residual_cosine_max_abs",
        "v10_zero_w_latent_max_abs",
        "v10_zero_w_output_max_abs",
        "v10_zero_z_output_max_abs",
        "v10_backend_cuda_matmul_allow_tf32",
        "v10_backend_cudnn_allow_tf32",
        "v10_backend_float32_matmul_precision_highest",
    )
    ledger = {name: float(metrics.get(name, float("inf"))) for name in names}
    nonfinite = {name: value for name, value in ledger.items() if not math.isfinite(value)}
    if nonfinite:
        raise RuntimeError(f"V10 geometry metrics contain nonfinite values: {nonfinite}")
    for name in ("v10_basis_orthogonality_max_abs", "v10_zc_analysis_max_abs"):
        if ledger[name] > 2.0e-5:
            raise RuntimeError(f"V10 exact analysis gate failed: {name}={ledger[name]:.6g}")
    if (
        ledger["v10_backend_cuda_matmul_allow_tf32"] != 0.0
        or ledger["v10_backend_cudnn_allow_tf32"] != 0.0
        or ledger["v10_backend_float32_matmul_precision_highest"] != 1.0
    ):
        raise RuntimeError(
            "V10 evaluator did not run under strict FP32 backend: "
            f"matmul_tf32={ledger['v10_backend_cuda_matmul_allow_tf32']} "
            f"cudnn_tf32={ledger['v10_backend_cudnn_allow_tf32']} "
            f"highest={ledger['v10_backend_float32_matmul_precision_highest']}"
        )
    if ledger["v10_basis_min_valid_column_coverage"] <= 1.0e-6:
        raise RuntimeError(
            "V10 fixed basis leaves a valid W coordinate unsupported: "
            f"min_coverage={ledger['v10_basis_min_valid_column_coverage']:.6g}"
        )
    if ledger["v10_residual_rowspace_max_abs"] > 2.0e-4:
        raise RuntimeError(
            "V10 orthogonal residual gate failed: "
            f"max_abs={ledger['v10_residual_rowspace_max_abs']:.6g}"
        )
    if ledger["v10_floor_residual_cosine_max_abs"] > 2.0e-5:
        raise RuntimeError(
            "V10 floor/residual orthogonality gate failed: "
            f"cos={ledger['v10_floor_residual_cosine_max_abs']:.6g}"
        )
    for name in (
        "v10_zero_w_latent_max_abs",
        "v10_zero_w_output_max_abs",
        "v10_zero_z_output_max_abs",
    ):
        if ledger[name] != 0.0:
            raise RuntimeError(f"V10 structural zero gate failed: {name}={ledger[name]:.6g}")
    return ledger


def validate_v10_numeric_telemetry_finite(metrics: Mapping[str, Any]) -> None:
    """Reject nonfinite scalar V10 telemetry before JSON persistence."""

    nonfinite = {
        name: float(value)
        for name, value in metrics.items()
        if name.startswith("v10_")
        and isinstance(value, (int, float))
        and not math.isfinite(float(value))
    }
    if nonfinite:
        raise RuntimeError(f"V10 numeric telemetry contains nonfinite values: {nonfinite}")


def validate_v10_step0_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Fail closed on V10 geometry and initialization-only floor quality."""

    ledger = validate_v10_geometry_metrics(metrics)
    init_names = (
        "v10_floor_mean_dir",
        "v10_floor_p95_dir",
        "v10_residual_to_floor_rms",
        "v10_residual_to_floor_rms_max",
        "v10_residual_to_floor_rms_p95",
        "matched_mean_dir",
        "matched_dir_p95",
    )
    init_ledger = {
        name: float(metrics.get(name, float("inf"))) for name in init_names
    }
    nonfinite = {
        name: value for name, value in init_ledger.items() if not math.isfinite(value)
    }
    if nonfinite:
        raise RuntimeError(f"V10 step-0 metrics contain nonfinite values: {nonfinite}")
    ledger.update(init_ledger)
    if ledger["v10_floor_mean_dir"] > 0.35 or ledger["v10_floor_p95_dir"] > 0.5:
        raise RuntimeError(
            "V10 step-0 protected floor is below the frozen quality gate: "
            f"mean={ledger['v10_floor_mean_dir']:.6g} p95={ledger['v10_floor_p95_dir']:.6g}"
        )
    if (
        not math.isfinite(ledger["v10_residual_to_floor_rms"])
        or not math.isfinite(ledger["v10_residual_to_floor_rms_max"])
        or not math.isfinite(ledger["v10_residual_to_floor_rms_p95"])
        or ledger["v10_residual_to_floor_rms_p95"] > 0.25
    ):
        raise RuntimeError(
            "V10 step-0 residual/floor energy exceeds the frozen p95<=0.25 bound: "
            f"global={ledger['v10_residual_to_floor_rms']:.6g} "
            f"p95={ledger['v10_residual_to_floor_rms_p95']:.6g} "
            f"max={ledger['v10_residual_to_floor_rms_max']:.6g}"
        )
    if (
        ledger["matched_mean_dir"] > ledger["v10_floor_mean_dir"] + 0.02
        or ledger["matched_dir_p95"] > ledger["v10_floor_p95_dir"] + 0.03
    ):
        raise RuntimeError(
            "V10 step-0 random residual degrades the protected floor beyond 0.03: "
            f"matched_mean={ledger['matched_mean_dir']:.6g} "
            f"floor_mean={ledger['v10_floor_mean_dir']:.6g} "
            f"matched_p95={ledger['matched_dir_p95']:.6g} "
            f"floor_p95={ledger['v10_floor_p95_dir']:.6g}"
        )
    return ledger


def validate_v10_step1_gradients(
    model: torch.nn.Module,
    *,
    adam_eps: float,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Require live V10 paths and record exact first-step Adam utilization."""

    raw_model = _raw_model(model)
    encoder = getattr(raw_model, "orthogonal_complement_encoder_v10", None)
    if encoder is None or len(encoder.experts) != 40:
        raise RuntimeError("V10 step-1 gradient gate requires exactly forty experts")
    groups: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def require_weight(key: str, module: torch.nn.Module) -> None:
        weight = getattr(module, "weight", None)
        grad = None if weight is None else weight.grad
        if not isinstance(weight, torch.nn.Parameter):
            failures.append(f"strategic parameter missing: {key}")
            groups[key] = {"state": "parameter_missing"}
            return
        if grad is None:
            failures.append(f"strategic gradient missing: {key}")
            groups[key] = {"state": "gradient_missing", "numel": int(weight.numel())}
            return
        if not bool(torch.isfinite(grad).all()):
            failures.append(f"strategic gradient nonfinite: {key}")
            groups[key] = {"state": "gradient_nonfinite", "numel": int(weight.numel())}
            return
        grad_float = grad.detach().float().reshape(-1)
        grad_abs = grad_float.abs()
        nonzero_count = int(torch.count_nonzero(grad_abs).item())
        if nonzero_count == 0:
            failures.append(f"strategic gradient identically zero: {key}")
        grad_quantiles = torch.quantile(
            grad_abs,
            torch.tensor((0.5, 0.9, 0.99, 0.999), device=grad_abs.device),
        )
        adam_u = grad_abs / (grad_abs + float(adam_eps))
        adam_u_quantiles = torch.quantile(
            adam_u,
            torch.tensor((0.05, 0.5, 0.95), device=adam_u.device),
        )
        adam_u_rms = float(adam_u.square().mean().sqrt().item())
        signal_delta = -float(learning_rate) * grad_float / (
            grad_abs + float(adam_eps)
        )
        weight_float = weight.detach().float().reshape(-1)
        signal_only_proposed = weight_float + signal_delta
        signal_only_parameter_change_fraction = float(
            (signal_only_proposed != weight_float).float().mean().item()
        )
        decayed = weight_float * (1.0 - float(learning_rate) * float(weight_decay))
        decay_delta = decayed - weight_float
        proposed = decayed + signal_delta
        total_delta = proposed - weight_float
        total_parameter_change_fraction = float(
            (proposed != weight_float).float().mean().item()
        )
        effective_update_rms = float(signal_delta.square().mean().sqrt().item())
        if effective_update_rms == 0.0:
            failures.append(f"strategic applied update identically zero: {key}")
        if signal_only_parameter_change_fraction == 0.0:
            failures.append(
                f"strategic signal-only FP32 parameter change identically zero: {key}"
            )
        weight_rms = float(weight_float.square().mean().sqrt().item())
        decay_update_rms = float(decay_delta.square().mean().sqrt().item())
        total_delta_rms = float(total_delta.square().mean().sqrt().item())
        groups[key] = {
            "state": "finite_nonzero" if nonzero_count > 0 else "finite_zero",
            "numel": int(weight.numel()),
            "grad_rms": float(grad_abs.square().mean().sqrt().item()),
            "grad_abs_p50": float(grad_quantiles[0].item()),
            "grad_abs_p90": float(grad_quantiles[1].item()),
            "grad_abs_p99": float(grad_quantiles[2].item()),
            "grad_abs_p999": float(grad_quantiles[3].item()),
            "grad_abs_max": float(grad_abs.max().item()),
            "grad_nonzero_fraction": nonzero_count / max(1, int(grad_abs.numel())),
            "adam_t1_u_rms": adam_u_rms,
            "adam_t1_u_p05": float(adam_u_quantiles[0].item()),
            "adam_t1_u_p50": float(adam_u_quantiles[1].item()),
            "adam_t1_u_p95": float(adam_u_quantiles[2].item()),
            "adam_t1_u_fraction_gt_0p01": float((adam_u > 0.01).float().mean().item()),
            "adam_t1_u_fraction_gt_0p1": float((adam_u > 0.1).float().mean().item()),
            "effective_update_rms": effective_update_rms,
            "weight_rms": weight_rms,
            "effective_update_to_weight_rms": (
                effective_update_rms / weight_rms if weight_rms > 0.0 else None
            ),
            "decay_update_rms": decay_update_rms,
            "total_delta_rms": total_delta_rms,
            "signal_only_fp32_parameter_change_fraction": (
                signal_only_parameter_change_fraction
            ),
            "total_fp32_parameter_change_fraction": total_parameter_change_fraction,
            "effective_update_to_decay_update_rms": (
                effective_update_rms / decay_update_rms
                if decay_update_rms > 0.0
                else None
            ),
            "total_delta_to_decay_update_rms": (
                total_delta_rms / decay_update_rms if decay_update_rms > 0.0 else None
            ),
        }

    for index, (expert, slot_mixer, code_projection) in enumerate(
        zip(encoder.experts, encoder.slot_mixers, encoder.code_projections, strict=True)
    ):
        for name, module in (
            ("query", expert.query_proj),
            ("weight_key", expert.weight_key_proj),
            ("weight_norm_key", expert.weight_norm_key_proj),
            ("context_key", expert.context_key_proj),
            ("raw_correction", expert.out_proj),
            ("ffn_in", expert.local_ffn[0]),
            ("ffn_out", expert.local_ffn[2]),
            ("slot_mixer", slot_mixer),
            ("code_projection", code_projection),
        ):
            require_weight(f"expert_{index:02d}.{name}", module)
    bridge = raw_model.mandatory_latent_bridge
    for name, module in (
        ("bridge.q", bridge.q_proj),
        ("bridge.k", bridge.k_proj),
        ("bridge.v", bridge.v_proj),
        ("bridge.out", bridge.out_proj),
        ("residual_head", raw_model.v10_residual_head),
    ):
        require_weight(name, module)
    trainable_parameter_count = 0
    trainable_parameter_numel = 0
    unused_parameter_names: list[str] = []
    unused_parameter_numel = 0
    zero_gradient_names: list[str] = []
    zero_gradient_numel = 0
    nonfinite_gradient_names: list[str] = []
    for name, parameter in raw_model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameter_count += 1
        trainable_parameter_numel += int(parameter.numel())
        if parameter.grad is None:
            unused_parameter_names.append(name)
            unused_parameter_numel += int(parameter.numel())
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite_gradient_names.append(name)
            continue
        if int(torch.count_nonzero(parameter.grad).item()) == 0:
            zero_gradient_names.append(name)
            zero_gradient_numel += int(parameter.numel())
    failures.extend(f"unused trainable parameter: {name}" for name in unused_parameter_names)
    failures.extend(
        f"nonfinite trainable gradient: {name}" for name in nonfinite_gradient_names
    )
    failures.extend(
        f"whole-tensor zero trainable gradient: {name}" for name in zero_gradient_names
    )
    complete_groups = [group for group in groups.values() if group.get("grad_rms") is not None]

    def minimum_group_metric(name: str) -> float | None:
        values = [float(group[name]) for group in complete_groups]
        return min(values) if values else None

    return {
        "schema": "weightclip_ae_v10_exact_b18_step1_gradients_v2",
        "adam_eps": float(adam_eps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "passed": not failures,
        "failures": failures,
        "minimum_group_grad_rms": minimum_group_metric("grad_rms"),
        "minimum_group_adam_t1_u_rms": minimum_group_metric("adam_t1_u_rms"),
        "minimum_group_effective_update_rms": minimum_group_metric(
            "effective_update_rms"
        ),
        "groups": groups,
        "strict_group_count": len(groups),
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_numel": trainable_parameter_numel,
        "unused_parameter_count": len(unused_parameter_names),
        "unused_parameter_numel": unused_parameter_numel,
        "unused_parameter_names": unused_parameter_names,
        "nonfinite_gradient_parameter_count": len(nonfinite_gradient_names),
        "nonfinite_gradient_parameter_names": nonfinite_gradient_names,
        "zero_gradient_parameter_count": len(zero_gradient_names),
        "zero_gradient_parameter_numel": zero_gradient_numel,
        "zero_gradient_parameter_fraction": (
            len(zero_gradient_names) / max(1, trainable_parameter_count)
        ),
        "zero_gradient_numel_fraction": (
            zero_gradient_numel / max(1, trainable_parameter_numel)
        ),
        "zero_gradient_parameter_names": zero_gradient_names,
    }


def _v10_expert_code_metrics(
    expert_codes: torch.Tensor,
    *,
    shuffled_expert_codes: torch.Tensor,
) -> dict[str, float]:
    """Bounded identity/diversity summaries for detached [E,N,L,D] codes."""

    if expert_codes.ndim != 4 or int(expert_codes.shape[0]) < 2:
        raise ValueError("V10 expert codes must be [E>=2,N,L,D]")
    codes = expert_codes.detach().float().cpu()
    shuffled_codes = shuffled_expert_codes.detach().float().cpu()
    if shuffled_codes.shape != codes.shape:
        raise ValueError("V10 fixed-X W-shuffle expert-code shape mismatch")
    metrics: dict[str, float] = {}
    for index, code in enumerate(codes):
        paired = shuffled_codes[index]
        pair_rms = (0.5 * (code.square().mean() + paired.square().mean())).sqrt()
        metrics[f"v10_expert_{index:02d}_code_rms"] = float(
            code.square().mean().sqrt().item()
        )
        metrics[f"v10_expert_{index:02d}_code_nonfinite_fraction"] = float(
            (~torch.isfinite(code)).float().mean().item()
        )
        metrics[f"v10_expert_{index:02d}_code_zero_fraction"] = float(
            (code == 0).float().mean().item()
        )
        metrics[f"v10_expert_{index:02d}_w_pair_shuffle_relative_delta"] = float(
            ((code - paired).square().mean().sqrt() / pair_rms.clamp_min(1.0e-24)).item()
        )
    flattened = codes.reshape(int(codes.shape[0]), -1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1.0e-24)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = cosine.masked_select(
        ~torch.eye(int(codes.shape[0]), dtype=torch.bool)
    ).abs()
    metrics["v10_expert_code_abs_cosine_offdiag_mean"] = float(off_diagonal.mean().item())
    metrics["v10_expert_code_abs_cosine_offdiag_max"] = float(off_diagonal.max().item())
    gram = centered @ centered.transpose(0, 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    trace = eigenvalues.sum()
    probabilities = eigenvalues / trace.clamp_min(1.0e-24)
    metrics["v10_expert_code_gram_stable_rank"] = float(
        (trace / eigenvalues.max().clamp_min(1.0e-24)).item()
    )
    metrics["v10_expert_code_gram_participation_rank"] = float(
        (trace.square() / eigenvalues.square().sum().clamp_min(1.0e-24)).item()
    )
    metrics["v10_expert_code_gram_effective_rank"] = float(
        torch.exp(-(probabilities * probabilities.clamp_min(1.0e-30).log()).sum()).item()
    )
    centered_norms = centered.norm(dim=1)
    centered_sum_norm = centered.sum(dim=0).norm()
    metrics["v10_expert_code_centered_resultant_over_rss"] = float(
        (
            centered_sum_norm
            / centered_norms.square().sum().sqrt().clamp_min(1.0e-24)
        ).item()
    )
    metrics["v10_expert_code_centered_coherence_ratio"] = float(
        (centered_sum_norm / centered_norms.sum().clamp_min(1.0e-24)).item()
    )
    raw_norms = flattened.norm(dim=1)
    raw_sum_norm = flattened.sum(dim=0).norm()
    metrics["v10_expert_code_raw_resultant_over_rss"] = float(
        (raw_sum_norm / raw_norms.square().sum().sqrt().clamp_min(1.0e-24)).item()
    )
    metrics["v10_expert_code_raw_coherence_ratio"] = float(
        (raw_sum_norm / raw_norms.sum().clamp_min(1.0e-24)).item()
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("V10 expert-code telemetry produced nonfinite values")
    return metrics


def _v10_family_leaveout_latent(
    latent: torch.Tensor,
    expert_codes: torch.Tensor,
    *,
    protected_dim: int,
    family_index: int,
) -> torch.Tensor:
    """Remove one of four interleaved expert families from za=sum(c_e)/sqrt(E)."""

    if expert_codes.ndim != 4:
        raise ValueError("V10 expert codes must be [E,N,L,D]")
    expert_count, batch_size, num_latents, adaptive_dim = expert_codes.shape
    shaped = latent.reshape(batch_size, num_latents, -1).float().clone()
    if int(shaped.shape[-1]) != int(protected_dim) + int(adaptive_dim):
        raise ValueError("V10 latent/expert-code dimensions do not match")
    indices = tuple(index for index in range(expert_count) if index % 4 == family_index)
    if not indices:
        raise ValueError(f"V10 family {family_index} has no experts")
    contribution = expert_codes[list(indices)].float().sum(dim=0) / math.sqrt(expert_count)
    shaped[..., protected_dim:] -= contribution
    return shaped


def _reset_v10_evaluator_capture(encoder: torch.nn.Module) -> None:
    encoder.capture_expert_states = False
    encoder.last_expert_codes = ()
    encoder.last_basis = None
    encoder.last_protected = None
    encoder.last_adaptive = None
    for expert in encoder.experts:
        expert.capture_routing_diagnostics = False
        expert.last_routing_diagnostics = None


def _v11_trunk_code_metrics(
    trunk_codes: torch.Tensor,
    *,
    shuffled_trunk_codes: torch.Tensor,
) -> dict[str, float]:
    """V11 trunk identity/diversity summaries for detached [4,N,L,16] codes."""

    if int(trunk_codes.shape[0]) != 4:
        raise ValueError("V11 requires exactly four captured trunk codes")
    base = _v10_expert_code_metrics(
        trunk_codes,
        shuffled_expert_codes=shuffled_trunk_codes,
    )
    metrics = {
        name.replace("v10_expert", "v11_trunk"): value
        for name, value in base.items()
    }
    for suffix in ("mean", "max"):
        old = f"v11_trunk_code_abs_cosine_offdiag_{suffix}"
        metrics[f"v11_trunk_code_coordinate_abs_cosine_offdiag_{suffix}"] = metrics.pop(old)
    codes = trunk_codes.detach().float().cpu()
    sample_grams: list[torch.Tensor] = []
    for code in codes:
        features = code.reshape(int(code.shape[0]), -1)
        features = features - features.mean(dim=0, keepdim=True)
        gram = features @ features.transpose(0, 1)
        sample_grams.append(gram / gram.norm().clamp_min(1.0e-24))
    cka_values = torch.stack(
        [
            (sample_grams[left] * sample_grams[right]).sum()
            for left in range(4)
            for right in range(left + 1, 4)
        ]
    )
    metrics["v11_trunk_sample_structure_linear_cka_mean"] = float(
        cka_values.mean().item()
    )
    metrics["v11_trunk_sample_structure_linear_cka_min"] = float(
        cka_values.min().item()
    )
    metrics["v11_trunk_sample_structure_linear_cka_max"] = float(
        cka_values.max().item()
    )
    return metrics


def _reset_v11_evaluator_capture(encoder: torch.nn.Module) -> None:
    encoder.capture_trunk_states = False
    encoder.last_trunk_codes = ()
    encoder.last_trunk_deltas = ()
    encoder.last_basis = None
    encoder.last_protected = None
    encoder.last_complement_patches = None
    encoder.last_complement_feed = None
    encoder.last_complement_feed_dtype = None
    encoder.last_adaptive = None
    for trunk in encoder.trunks:
        trunk.capture_states = False
        trunk.retain_code_gradient = False
        trunk.last_delta = None
        trunk.last_code = None
        trunk.last_code_for_gradient = None
        trunk.code_tensors_for_gradient.clear()
        for block in trunk.blocks:
            block.capture_routing_diagnostics = False
            block.last_routing_diagnostics = None


def _v11_metrics_as_v10(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        ("v10_" + name[4:] if name.startswith("v11_") else name): value
        for name, value in metrics.items()
    }


def validate_v11_geometry_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Fail closed on the V10 floor contract plus V11 complement-only input."""

    v10_ledger = validate_v10_geometry_metrics(_v11_metrics_as_v10(metrics))
    ledger = {"v11_" + name[4:]: value for name, value in v10_ledger.items()}
    extra_names = (
        "v11_complement_input_rowspace_max_abs",
        "v11_complement_feed_is_bfloat16",
        "v11_complement_feed_cast_relative_rms",
        "v11_complement_feed_rowspace_relative_rms",
        "v11_native_latent_capture_max_abs",
        "v11_manual_decode_max_abs",
        "v11_manual_decode_relative_rms",
    )
    extras = {name: float(metrics.get(name, float("inf"))) for name in extra_names}
    if not all(math.isfinite(value) for value in extras.values()):
        raise RuntimeError(f"V11 complement geometry contains nonfinite values: {extras}")
    if extras["v11_complement_input_rowspace_max_abs"] > 2.0e-4:
        raise RuntimeError(
            "V11 trunk input is not in the protected orthogonal complement: "
            f"max_abs={extras['v11_complement_input_rowspace_max_abs']:.6g}"
        )
    if extras["v11_complement_feed_is_bfloat16"] != 1.0:
        raise RuntimeError(
            "V11 production trunk feed must be BF16 under AMP; "
            f"observed={extras['v11_complement_feed_is_bfloat16']}"
        )
    if extras["v11_complement_feed_cast_relative_rms"] >= 0.01:
        raise RuntimeError(
            "V11 BF16 complement feed has excessive quantization error: "
            f"rel={extras['v11_complement_feed_cast_relative_rms']:.6g}"
        )
    if extras["v11_complement_feed_rowspace_relative_rms"] >= 0.01:
        raise RuntimeError(
            "V11 BF16 complement feed leaked excessively into protected rowspace: "
            f"rel={extras['v11_complement_feed_rowspace_relative_rms']:.6g}"
        )
    for name in (
        "v11_native_latent_capture_max_abs",
        "v11_manual_decode_max_abs",
        "v11_manual_decode_relative_rms",
    ):
        if extras[name] > 1.0e-6:
            raise RuntimeError(f"V11 native/manual path parity failed: {name}={extras[name]:.6g}")
    ledger.update(extras)
    return ledger


def validate_v11_step0_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """V11 step-0 floor/energy gate plus four nondegenerate trunk codes."""

    v10_ledger = validate_v10_step0_metrics(_v11_metrics_as_v10(metrics))
    ledger = {"v11_" + name[4:] if name.startswith("v10_") else name: value for name, value in v10_ledger.items()}
    ledger.update(validate_v11_geometry_metrics(metrics))
    for trunk_index in range(4):
        name = f"v11_trunk_{trunk_index:02d}_code_rms"
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"V11 step-0 trunk code is dead/nonfinite: {name}={value}")
        ledger[name] = value
        delta_name = f"v11_trunk_{trunk_index:02d}_delta_rms"
        delta_value = float(metrics.get(delta_name, float("nan")))
        if not math.isfinite(delta_value) or delta_value <= 0.0:
            raise RuntimeError(
                f"V11 step-0 trunk raw delta is dead/nonfinite: {delta_name}={delta_value}"
            )
        ledger[delta_name] = delta_value
        identity_name = f"v11_trunk_{trunk_index:02d}_w_pair_shuffle_relative_delta"
        identity_value = float(metrics.get(identity_name, float("nan")))
        if not math.isfinite(identity_value) or identity_value < 0.1:
            raise RuntimeError(
                f"V11 step-0 trunk lacks W-specific identity: {identity_name}={identity_value}"
            )
        ledger[identity_name] = identity_value
    return ledger


def validate_v11_numeric_telemetry_finite(metrics: Mapping[str, Any]) -> None:
    nonfinite = {
        name: float(value)
        for name, value in metrics.items()
        if (name.startswith("v11_") or "_v11_" in name)
        and isinstance(value, (int, float))
        and not math.isfinite(float(value))
    }
    if nonfinite:
        raise RuntimeError(f"V11 numeric telemetry contains nonfinite values: {nonfinite}")


def v11_scientific_contract_failures(
    metrics: Mapping[str, Any],
    *,
    step0_complement_normalized_mse: float,
    final: bool,
) -> list[str]:
    """Precommitted V11 progress/identity checks; final targets are nonfatal."""

    failures: list[str] = []
    complement = float(
        metrics.get("v11_complement_normalized_mse_mean", float("inf"))
    )
    required_mse_fraction = 0.7 if final else 0.8
    if complement > required_mse_fraction * float(step0_complement_normalized_mse):
        failures.append(
            f"complement_mse_fraction>{required_mse_fraction}: "
            f"step0_mse={step0_complement_normalized_mse:.6g} current_mse={complement:.6g}"
        )
    if float(metrics.get("matched_mean_dir", float("inf"))) >= float(
        metrics.get("v11_floor_mean_dir", float("-inf"))
    ):
        failures.append("matched_mean_dir_not_below_floor")
    code_rms: list[float] = []
    for trunk_index in range(4):
        identity = float(
            metrics.get(
                f"v11_trunk_{trunk_index:02d}_w_pair_shuffle_relative_delta",
                float("-inf"),
            )
        )
        if identity < 0.1:
            failures.append(f"trunk_{trunk_index:02d}_w_identity_rel={identity:.6g}<0.1")
        code = float(
            metrics.get(f"v11_trunk_{trunk_index:02d}_code_rms", float("nan"))
        )
        delta = float(
            metrics.get(f"v11_trunk_{trunk_index:02d}_delta_rms", float("nan"))
        )
        if not math.isfinite(code) or code <= 0.0:
            failures.append(f"trunk_{trunk_index:02d}_code_dead_or_nonfinite")
        if not math.isfinite(delta) or delta <= 0.0:
            failures.append(f"trunk_{trunk_index:02d}_delta_dead_or_nonfinite")
        code_rms.append(code)
    if final:
        if float(metrics.get("matched_mean_dir", float("inf"))) > 0.22246:
            failures.append("matched_mean_dir>0.22246")
        if float(metrics.get("matched_dir_p95", float("inf"))) > 0.22516:
            failures.append("matched_dir_p95>0.22516")
        if float(metrics.get("v11_za_shuffle_min_dir_delta", float("-inf"))) < 0.005:
            failures.append("za_shuffle_min_dir_delta<0.005")
        if float(metrics.get("v11_za_shuffle_positive_fraction", 0.0)) < 1.0:
            failures.append("za_shuffle_positive_fraction<1.0")
        for trunk_index in range(4):
            if float(
                metrics.get(
                    f"v11_trunk_{trunk_index:02d}_leaveout_dir_delta_mean",
                    float("-inf"),
                )
            ) <= 0.0:
                failures.append(f"trunk_{trunk_index:02d}_leaveout_mean<=0")
        finite_codes = [value for value in code_rms if math.isfinite(value)]
        if len(finite_codes) == 4:
            median = float(torch.tensor(finite_codes).median().item())
            for trunk_index, value in enumerate(code_rms):
                if value < 0.1 * median:
                    failures.append(
                        f"trunk_{trunk_index:02d}_code_rms_below_0.1x_median"
                    )
    return failures


def validate_v11_step1_gradients(
    model: torch.nn.Module,
    *,
    adam_eps: float,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Reuse the complete V10 gradient inventory for V11's forty direct blocks."""

    raw_model = _raw_model(model)
    encoder = getattr(raw_model, "four_trunk_complement_encoder_v11", None)
    if encoder is None or len(encoder.trunks) != 4:
        raise RuntimeError("V11 step-1 gate requires exactly four trunks")
    blocks: list[torch.nn.Module] = []
    mixers: list[torch.nn.Module] = []
    projections: list[torch.nn.Module] = []
    for trunk in encoder.trunks:
        if len(trunk.blocks) != 10:
            raise RuntimeError("V11 step-1 gate requires ten blocks per trunk")
        for block in trunk.blocks:
            blocks.append(block)
            # Repetition is intentional: the complete global inventory below is
            # authoritative, while these strategic rows expose each trunk's
            # terminal short path next to every one of its ten routing blocks.
            mixers.append(trunk.slot_mixer)
            projections.append(trunk.code_projection)
    proxy = SimpleNamespace(
        orthogonal_complement_encoder_v10=SimpleNamespace(
            experts=blocks,
            slot_mixers=mixers,
            code_projections=projections,
        ),
        mandatory_latent_bridge=raw_model.mandatory_latent_bridge,
        v10_residual_head=raw_model.v11_residual_head,
        named_parameters=raw_model.named_parameters,
    )
    ledger = validate_v10_step1_gradients(
        proxy,
        adam_eps=adam_eps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    renamed_groups: dict[str, Any] = {}
    for name, value in ledger["groups"].items():
        if name.startswith("expert_"):
            block_index = int(name[7:9])
            suffix = name[9:]
            if suffix in {".slot_mixer", ".code_projection"}:
                if block_index % 10 != 0:
                    continue
                name = f"trunk_{block_index // 10:02d}{suffix}"
            else:
                name = f"trunk_{block_index // 10:02d}.block_{block_index % 10:02d}{suffix}"
        elif name == "residual_head":
            name = "v11_residual_head"
        renamed_groups[name] = value
    ledger["schema"] = "weightclip_ae_v11_exact_b18_step1_gradients_v1"
    ledger["groups"] = renamed_groups
    ledger["strict_group_count"] = len(renamed_groups)
    if len(renamed_groups) != 293:
        ledger["failures"].append(
            f"V11 strategic gradient inventory expected 293 groups, got {len(renamed_groups)}"
        )
    complete_groups = [
        group for group in renamed_groups.values() if group.get("grad_rms") is not None
    ]
    for output_name, group_name in (
        ("minimum_group_grad_rms", "grad_rms"),
        ("minimum_group_adam_t1_u_rms", "adam_t1_u_rms"),
        ("minimum_group_effective_update_rms", "effective_update_rms"),
    ):
        values = [float(group[group_name]) for group in complete_groups]
        ledger[output_name] = min(values) if values else None
    if (
        int(ledger["trainable_parameter_count"]) != 474
        or int(ledger["trainable_parameter_numel"]) != 235_073_152
    ):
        ledger["failures"].append(
            "V11 trainable topology drift: "
            f"tensors={ledger['trainable_parameter_count']} "
            f"numel={ledger['trainable_parameter_numel']}"
        )
    code_grads: list[torch.Tensor] = []
    for trunk_index, trunk in enumerate(encoder.trunks):
        tensors = tuple(trunk.code_tensors_for_gradient)
        grads = [tensor.grad for tensor in tensors]
        if (
            not grads
            or any(grad is None for grad in grads)
            or not all(
                bool(torch.isfinite(grad).all())
                for grad in grads
                if grad is not None
            )
        ):
            ledger["failures"].append(
                f"V11 serialized trunk {trunk_index} gradient missing/nonfinite"
            )
            continue
        grad_f = torch.cat(
            [grad.detach().float().reshape(-1) for grad in grads if grad is not None]
        )
        rms = float(grad_f.square().mean().sqrt().item())
        ledger[f"trunk_{trunk_index:02d}_serialized_code_grad_rms"] = rms
        if rms == 0.0:
            ledger["failures"].append(
                f"V11 serialized trunk {trunk_index} gradient identically zero"
            )
        code_grads.append(grad_f)
    if len(code_grads) == 4:
        normalized = torch.stack(
            [grad / grad.norm().clamp_min(1.0e-30) for grad in code_grads]
        )
        cosine = normalized @ normalized.transpose(0, 1)
        offdiag = cosine.masked_select(~torch.eye(4, dtype=torch.bool, device=cosine.device))
        max_abs_cosine = float(offdiag.abs().max().item())
        ledger["serialized_trunk_code_grad_abs_cosine_max"] = max_abs_cosine
        if max_abs_cosine >= 0.999:
            ledger["failures"].append(
                "V11 serialized trunk gradients collapsed to a common direction: "
                f"max_abs_cosine={max_abs_cosine:.6g}"
            )
    for projection_name in ("k_proj", "v_proj"):
        weight = getattr(raw_model.mandatory_latent_bridge, projection_name).weight
        grad = weight.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            ledger["failures"].append(
                f"V11 bridge {projection_name} column gradients missing/nonfinite"
            )
            continue
        for trunk_index in range(4):
            column_grad = grad[:, trunk_index * 16 : (trunk_index + 1) * 16].float()
            rms = float(column_grad.square().mean().sqrt().item())
            ledger[
                f"bridge_{projection_name}_trunk_{trunk_index:02d}_column_grad_rms"
            ] = rms
            if rms == 0.0:
                ledger["failures"].append(
                    f"V11 bridge {projection_name} trunk {trunk_index} columns are dead"
                )
    ledger["passed"] = not ledger["failures"]
    for trunk in encoder.trunks:
        trunk.retain_code_gradient = False
        trunk.last_code_for_gradient = None
        trunk.code_tensors_for_gradient.clear()
    return ledger


class CanonicalOperatorSetTrainingDataset(OperatorBankTrainingDataset):
    """Consumer-only fixed canonical full-operator cycle over an exact key set."""

    def __init__(
        self,
        pair_manifest: str | Path,
        *,
        selected_operators: Sequence[Mapping[str, str]],
        expected_operator_count: int,
        seed: int,
        hot_shards: int,
        expected_pair_manifest_sha256: str,
        expected_selection_sha256: str,
        expected_schedule_sha256: str,
    ) -> None:
        super().__init__(
            pair_manifest,
            seed=seed,
            repeat=True,
            permutation_views=False,
            canonical_probability=1.0,
            hot_shards=hot_shards,
            expected_pair_manifest_sha256=expected_pair_manifest_sha256,
            rank=0,
            world_size=1,
        )
        keys = tuple(
            (str(row.get("checkpoint_sha256", "")), str(row.get("layer_key", "")))
            for row in selected_operators
        )
        if len(keys) != int(expected_operator_count) or len(set(keys)) != len(keys):
            raise ValueError(
                f"canonical operator set requires {expected_operator_count} unique identities"
            )
        missing = [key for key in keys if key not in self.operator_groups]
        if missing:
            raise ValueError(f"selected operator identities are absent from the immutable bank: {missing}")
        self.operator_groups = {key: self.operator_groups[key] for key in keys}
        self._keys = list(keys)
        self._rounds = self._round_robin_rounds(self._keys, seed=int(seed))
        self.training_rounds = tuple(self._rounds[:62])
        self.eval_derangement_round = tuple(self._rounds[62])
        self.match_schedule = tuple(pair for round_pairs in self.training_rounds for pair in round_pairs)
        self.selection_sha256 = self._selection_sha256(self._keys)
        self.schedule_sha256 = self._schedule_sha256(self._rounds)
        if self.selection_sha256 != str(expected_selection_sha256):
            raise ValueError("V9 exact64 selected-operator digest mismatch")
        if self.schedule_sha256 != str(expected_schedule_sha256):
            raise ValueError("V9 exact64 round-robin schedule digest mismatch")
        self._records_per_cycle = sum(
            len(self.operator_groups[left]) + len(self.operator_groups[right])
            for left, right in self.match_schedule
        )
        self._cycle_cache.clear()
        self._group_cache.clear()
        self._locality_schedule_cache.clear()

        reference = self.operator_groups[self._keys[0]][0].metadata["operator"]
        reference_layer = self._keys[0][1]
        reference_shape = tuple(reference["matrix_shape"])
        reference_tiles = len(self.operator_groups[self._keys[0]])
        datasets: Counter[str] = Counter()
        lineages: set[tuple[str, str]] = set()
        epochs: Counter[int] = Counter()
        for key in self._keys:
            locations = self.operator_groups[key]
            row_metadata = locations[0].metadata
            metadata = row_metadata["operator"]
            if (
                key[1] != reference_layer
                or tuple(metadata["matrix_shape"]) != reference_shape
                or len(locations) != reference_tiles
            ):
                raise ValueError(
                    "canonical operator set requires one layer, matrix shape, and tile count"
                )
            rows, cols = map(int, reference_shape)
            expected_starts = {
                (row, col)
                for row in range(0, rows, 128)
                for col in range(0, cols, 128)
            }
            actual_starts = {
                (
                    int(location.metadata["tile"]["row_start"]),
                    int(location.metadata["tile"]["col_start"]),
                )
                for location in locations
            }
            if actual_starts != expected_starts or len(actual_starts) != len(locations):
                raise ValueError(f"selected operator {key} lacks exact canonical tile coverage")
            dataset = str(row_metadata["dataset"])
            lineage = str(row_metadata["lineage_id"])
            epoch = int(row_metadata["checkpoint_index_zero_based"])
            datasets[dataset] += 1
            lineage_key = (dataset, lineage)
            if lineage_key in lineages:
                raise ValueError(f"V9 exact64 selection repeats lineage {lineage_key!r}")
            lineages.add(lineage_key)
            epochs[epoch] += 1
        if len(datasets) != 10 or sorted(datasets.values()) != [6] * 6 + [7] * 4:
            raise ValueError(f"V9 exact64 dataset quota mismatch: {dict(datasets)}")
        if epochs != Counter({43: 32, 44: 32}):
            raise ValueError(f"V9 exact64 epoch balance mismatch: {dict(epochs)}")

    @staticmethod
    def _selection_sha256(keys: Sequence[tuple[str, str]]) -> str:
        body = json.dumps(list(keys), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @staticmethod
    def _schedule_sha256(
        rounds: Sequence[Sequence[tuple[tuple[str, str], tuple[str, str]]]],
    ) -> str:
        body = json.dumps(list(rounds), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @staticmethod
    def _round_robin_rounds(
        keys: Sequence[tuple[str, str]],
        *,
        seed: int,
    ) -> tuple[tuple[tuple[tuple[str, str], tuple[str, str]], ...], ...]:
        if len(keys) != 64:
            raise ValueError("the frozen V9 diagnostic schedule requires exactly 64 teams")
        teams = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{int(seed)}|{key[0]}|{key[1]}".encode("utf-8")
            ).hexdigest(),
        )
        rounds: list[tuple[tuple[tuple[str, str], tuple[str, str]], ...]] = []
        for _round_index in range(63):
            rounds.append(
                tuple((teams[index], teams[-1 - index]) for index in range(len(teams) // 2))
            )
            teams = [teams[0], teams[-1], *teams[1:-1]]
        if len({tuple(sorted(pair)) for round_pairs in rounds for pair in round_pairs}) != 2016:
            raise RuntimeError("64-team round robin did not produce every unordered pair exactly once")
        return tuple(rounds)

    @property
    def exact_operator_cycle(self) -> bool:
        return True

    def _cycle_plan(
        self,
        cycle: int,
    ) -> tuple[list[tuple[tuple[str, str], int]], dict[str, dict[str, Any] | None]]:
        cached = self._cycle_cache.get(int(cycle))
        if cached is not None:
            self._cycle_cache.move_to_end(int(cycle))
            return cached
        plan = [
            (key, local_index)
            for left, right in self.match_schedule
            for local_index in range(len(self.operator_groups[left]))
            for key in (left, right)
        ]
        expected_tiles = {
            (key, local_index)
            for key in self._keys
            for local_index in range(len(self.operator_groups[key]))
        }
        multiplicities = Counter(plan)
        if (
            len(plan) != self._records_per_cycle
            or set(multiplicities) != expected_tiles
            or set(multiplicities.values()) != {62}
        ):
            raise RuntimeError(
                "canonical operator training cycle must expose every selected tile exactly "
                "once per one of the 62 scheduled matches"
            )
        result = (plan, {checkpoint: None for checkpoint, _layer in self._keys})
        self._cycle_cache[int(cycle)] = result
        while len(self._cycle_cache) > 2:
            self._cycle_cache.popitem(last=False)
        return result

    def locality_plan(self, cycle: int) -> tuple[tuple[tuple[str, str], int], ...]:
        plan, _views = self._cycle_plan(int(cycle))
        return tuple(plan)


@contextmanager
def canonical_operator_set_data_pipeline(
    pair_manifest: str | Path,
    *,
    selected_operators: Sequence[Mapping[str, str]],
    expected_operator_count: int,
    seed: int,
    hot_shards: int,
    expected_pair_manifest_sha256: str,
    expected_selection_sha256: str,
    expected_schedule_sha256: str,
    max_active_strata: int,
    max_active_bundle_bytes: int,
) -> Iterator[tuple[OperatorBankBundleDataset, CommittedBundleSampler]]:
    source = CanonicalOperatorSetTrainingDataset(
        pair_manifest,
        selected_operators=selected_operators,
        expected_operator_count=expected_operator_count,
        seed=seed,
        hot_shards=hot_shards,
        expected_pair_manifest_sha256=expected_pair_manifest_sha256,
        expected_selection_sha256=expected_selection_sha256,
        expected_schedule_sha256=expected_schedule_sha256,
    )
    dataset = OperatorBankBundleDataset(
        source,
        max_active_strata=max_active_strata,
        max_active_bundle_bytes=max_active_bundle_bytes,
    )
    yield dataset, CommittedBundleSampler(dataset)


class OperatorSetOverfitEvaluator:
    """Full-set replay with per-operator matched and cyclic-shuffle controls."""

    def __init__(
        self,
        *,
        source: CanonicalOperatorSetTrainingDataset,
        output_path: str | Path,
        every_steps: int,
        patch_size: int,
        gamma: float,
        lambda_dir: float,
        lambda_scale: float,
        huber_delta: float,
    ) -> None:
        if not source.exact_operator_cycle or len(source._keys) < 2:
            raise ValueError("operator-set evaluator requires at least two fixed-cycle operators")
        self.source = source
        self.output_path = Path(output_path)
        self.every_steps = int(every_steps)
        self.patch_size = int(patch_size)
        self.gamma = float(gamma)
        self.lambda_dir = float(lambda_dir)
        self.lambda_scale = float(lambda_scale)
        self.huber_delta = float(huber_delta)
        self.v11_step0_complement_normalized_mse: float | None = None

    def should_run(self, step: int, *, final_step: int) -> bool:
        return int(step) == int(final_step) or int(step) % self.every_steps == 0

    def _evaluation_assignments(self) -> list[tuple[tuple[str, str], int]]:
        tile_count = len(self.source.operator_groups[self.source._keys[0]])
        assignments = [
            (key, local_index)
            for local_index in range(tile_count)
            for key in self.source._keys
        ]
        if len(assignments) != 64 * tile_count or len(set(assignments)) != len(assignments):
            raise RuntimeError("full operator-set evaluation plan is incomplete")
        return assignments

    def _materialize_evaluation_microbatches(
        self,
        *,
        device: torch.device,
        batch_size: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]]:
        assignments = self._evaluation_assignments()
        samples = {
            key: self.source._materialize_group(key, None)
            for key in self.source._keys
        }
        rows: list[Any] = [samples[key][local_index] for key, local_index in assignments]
        result = []
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            indices = tuple(range(start, start + len(chunk)))
            result.append(
                (
                    torch.stack([row.weight for row in chunk]),
                    torch.stack([row.x for row in chunk]),
                    torch.stack([row.meta["x_mask"] for row in chunk]),
                    torch.stack([row.meta["d_in_mask"] for row in chunk]),
                    torch.stack([row.meta["d_out_mask"] for row in chunk]),
                    indices,
                )
            )
        return result

    @torch.no_grad()
    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        step: int,
        microbatches: Sequence[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Sequence[int],
            ]
        ] | None = None,
        eval_batch_size: int = 8,
    ) -> dict[str, Any]:
        raw_model = _raw_model(model)
        was_training = raw_model.training
        raw_model.eval()
        refiner = getattr(raw_model, "hybrid_content_readout_v9", None)
        v10_encoder = getattr(raw_model, "orthogonal_complement_encoder_v10", None)
        v10_enabled = bool(getattr(raw_model, "use_orthogonal_complement_v10", False))
        v11_encoder = getattr(raw_model, "four_trunk_complement_encoder_v11", None)
        v11_enabled = bool(getattr(raw_model, "use_four_trunk_complement_v11", False))
        orthogonal_encoder = v11_encoder if v11_enabled else v10_encoder
        orthogonal_enabled = v10_enabled or v11_enabled
        if refiner is not None:
            refiner.capture_stage_states = True
            for block in refiner.blocks:
                block.capture_routing_diagnostics = True
        if v10_encoder is not None and v10_enabled:
            v10_encoder.capture_expert_states = True
            for expert in v10_encoder.experts:
                expert.capture_routing_diagnostics = True
        if v11_encoder is not None and v11_enabled:
            v11_encoder.capture_trunk_states = True
            for block in v11_encoder.routing_blocks:
                block.capture_routing_diagnostics = True
        try:
            device = next(raw_model.parameters()).device
            microbatches = self._materialize_evaluation_microbatches(
                device=device,
                batch_size=int(eval_batch_size),
            )
            assignments = self._evaluation_assignments()
            lookup = {assignment: index for index, assignment in enumerate(assignments)}
            target_cpu = torch.cat([row[0] for row in microbatches], dim=0)
            x_cpu = torch.cat([row[1] for row in microbatches], dim=0)
            x_mask_cpu = torch.cat([row[2] for row in microbatches], dim=0)
            d_in_mask_cpu = torch.cat([row[3] for row in microbatches], dim=0)
            d_out_mask_cpu = torch.cat([row[4] for row in microbatches], dim=0)

            matched_chunks: list[torch.Tensor] = []
            latent_chunks: list[torch.Tensor] = []
            carrier_chunks: list[torch.Tensor] = []
            adaptive_chunks: list[torch.Tensor] = []
            zero_w_chunks: list[torch.Tensor] = []
            zero_w_latent_chunks: list[torch.Tensor] = []
            zero_z_chunks: list[torch.Tensor] = []
            v10_floor_chunks: list[torch.Tensor] = []
            v10_orthogonality_errors: list[float] = []
            v10_analysis_errors: list[float] = []
            v10_residual_rowspace_errors: list[float] = []
            v10_floor_residual_inner_products: list[float] = []
            v10_floor_residual_cosines: list[float] = []
            v10_basis_min_valid_column_coverages: list[float] = []
            v10_floor_sumsq = 0.0
            v10_residual_sumsq = 0.0
            v10_residual_to_floor_rms_max = 0.0
            replay: list[tuple[torch.Tensor, ...]] = []
            stage_chunks: list[list[torch.Tensor]] | None = None
            routing_rows: list[list[dict[str, float]]] = []
            v10_expert_code_chunks: list[list[torch.Tensor]] | None = None
            v10_fixed_x_w_expert_code_chunks: list[list[torch.Tensor]] | None = None
            v10_routing_rows: list[list[dict[str, float]]] = []
            v11_trunk_code_chunks: list[list[torch.Tensor]] | None = None
            v11_trunk_delta_chunks: list[list[torch.Tensor]] | None = None
            v11_fixed_x_w_trunk_code_chunks: list[list[torch.Tensor]] | None = None
            v11_routing_rows: list[list[dict[str, float]]] = []
            v11_complement_rowspace_errors: list[float] = []
            v11_complement_feed_bfloat16: list[float] = []
            v11_complement_feed_cast_errors: list[float] = []
            v11_complement_feed_rowspace_relative: list[float] = []
            v11_latent_capture_errors: list[float] = []
            v11_manual_decode_errors: list[float] = []
            v11_manual_decode_relative_errors: list[float] = []
            for W_cpu, x_cpu_chunk, x_mask_cpu_chunk, d_in_cpu, d_out_cpu, _indices in microbatches:
                if refiner is not None:
                    refiner.capture_stage_states = True
                    for block in refiner.blocks:
                        block.capture_routing_diagnostics = True
                if v11_encoder is not None and v11_enabled:
                    for block in v11_encoder.routing_blocks:
                        block.capture_routing_diagnostics = True
                W = W_cpu.to(device)
                x = x_cpu_chunk.to(device)
                x_mask = x_mask_cpu_chunk.to(device)
                d_in_mask = d_in_cpu.to(device)
                d_out_mask = d_out_cpu.to(device)
                W_hat, _mu, _logvar, _pred_dirs, debug = raw_model.forward_debug(
                    W,
                    x,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    disable_z_shortcut=True,
                )
                matched_chunks.append(W_hat.detach().cpu())
                latent_chunks.append(debug["latent_decoder_z"].detach().cpu())
                if orthogonal_enabled:
                    if orthogonal_encoder is None:
                        raise RuntimeError("V10/V11 evaluator lost its encoder")
                    latent = debug["latent_decoder_z"].view(
                        int(W.shape[0]),
                        int(raw_model.cfg.big_vae.num_latents),
                        int(raw_model.cfg.big_vae.d_lat),
                    )
                    if v10_enabled:
                        assert v10_encoder is not None
                        if len(v10_encoder.last_expert_codes) != len(v10_encoder.experts):
                            raise RuntimeError("V10 evaluator did not capture every expert code")
                        if v10_expert_code_chunks is None:
                            v10_expert_code_chunks = [
                                [] for _ in v10_encoder.last_expert_codes
                            ]
                        for expert_index, code in enumerate(v10_encoder.last_expert_codes):
                            v10_expert_code_chunks[expert_index].append(code.cpu())
                        chunk_routing: list[dict[str, float]] = []
                        for expert in v10_encoder.experts:
                            diagnostics = expert.last_routing_diagnostics
                            if diagnostics is None:
                                raise RuntimeError("V10 routing diagnostics were not captured")
                            chunk_routing.append(
                                {
                                    name: float(value.item())
                                    for name, value in diagnostics.items()
                                    if name != "anchor_family_id"
                                }
                            )
                        v10_routing_rows.append(chunk_routing)
                    if v11_enabled:
                        assert v11_encoder is not None
                        if len(v11_encoder.last_trunk_codes) != len(v11_encoder.trunks):
                            raise RuntimeError("V11 evaluator did not capture every trunk code")
                        if v11_trunk_code_chunks is None:
                            v11_trunk_code_chunks = [
                                [] for _ in v11_encoder.last_trunk_codes
                            ]
                        for trunk_index, code in enumerate(v11_encoder.last_trunk_codes):
                            v11_trunk_code_chunks[trunk_index].append(code.cpu())
                        if len(v11_encoder.last_trunk_deltas) != len(v11_encoder.trunks):
                            raise RuntimeError("V11 evaluator did not capture every trunk delta")
                        if v11_trunk_delta_chunks is None:
                            v11_trunk_delta_chunks = [
                                [] for _ in v11_encoder.last_trunk_deltas
                            ]
                        for trunk_index, delta in enumerate(v11_encoder.last_trunk_deltas):
                            v11_trunk_delta_chunks[trunk_index].append(delta.cpu())
                        captured_latent = torch.cat(
                            [v11_encoder.last_protected, *v11_encoder.last_trunk_codes],
                            dim=-1,
                        ).reshape_as(debug["latent_decoder_z"])
                        v11_latent_capture_errors.append(
                            float(
                                (
                                    captured_latent.float()
                                    - debug["latent_decoder_z"].float()
                                )
                                .abs()
                                .amax()
                                .item()
                            )
                        )
                        chunk_routing = []
                        for block in v11_encoder.routing_blocks:
                            diagnostics = block.last_routing_diagnostics
                            if diagnostics is None:
                                raise RuntimeError("V11 routing diagnostics were not captured")
                            chunk_routing.append(
                                {
                                    name: float(value.item())
                                    for name, value in diagnostics.items()
                                    if name != "anchor_family_id"
                                }
                            )
                        v11_routing_rows.append(chunk_routing)
                        for block in v11_encoder.routing_blocks:
                            block.capture_routing_diagnostics = False
                            block.last_routing_diagnostics = None
                    protected_dim = int(raw_model.cfg.big_vae.v10_protected_dim)
                    floor_z = latent.clone()
                    floor_z[..., protected_dim:] = 0
                    floor_hat, *_rest = raw_model._decode_from_decoder_latent(
                        floor_z,
                        dist_patch_by_patch=debug["dist_patch_by_patch"],
                        patch_mask=debug["patch_mask"],
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                        d_in=int(W.shape[1]),
                        d_out=int(W.shape[2]),
                        d_in_pad=int(debug["d_in_pad"]),
                        T=int(debug["T"]),
                        disable_z_shortcut=True,
                    )
                    v10_floor_chunks.append(floor_hat.detach().cpu())
                    with torch.autocast(device_type=W.device.type, enabled=False):
                        basis = orthogonal_encoder.basis(
                            debug["patch_mask"], d_out_mask
                        ).float()
                        gram = torch.einsum("brs,bks->brk", basis, basis)
                        identity = torch.eye(
                            basis.shape[1], device=basis.device, dtype=basis.dtype
                        ).unsqueeze(0)
                        v10_orthogonality_errors.append(
                            float((gram - identity).abs().amax().item())
                        )
                        key_valid = (
                            d_out_mask.unsqueeze(-1) & debug["patch_mask"].unsqueeze(1)
                        ).reshape(int(W.shape[0]), -1)
                        coverage = basis.square().sum(dim=1)
                        v10_basis_min_valid_column_coverages.append(
                            float(coverage.masked_select(key_valid).amin().item())
                        )
                        p = int(raw_model.cfg.patch_size)
                        W_pad = torch.zeros(
                            int(W.shape[0]),
                            int(debug["d_in_pad"]),
                            int(W.shape[2]),
                            device=W.device,
                            dtype=torch.float32,
                        )
                        W_pad[:, : int(W.shape[1]), :] = W.float()
                        w_patches = W_pad.transpose(1, 2).reshape(
                            int(W.shape[0]), int(W.shape[2]) * int(debug["T"]), p
                        )
                        expected_protected = torch.einsum("brs,bsp->brp", basis, w_patches)
                        actual_protected = latent[..., :protected_dim].reshape_as(
                            expected_protected
                        ).float()
                        v10_analysis_errors.append(
                            float((actual_protected - expected_protected).abs().amax().item())
                        )
                        if v11_enabled:
                            assert v11_encoder is not None
                            complement = v11_encoder.last_complement_patches
                            complement_feed = v11_encoder.last_complement_feed
                            if complement is None or complement_feed is None:
                                raise RuntimeError("V11 matched complement capture is missing")
                            complement_flat = complement.reshape_as(w_patches).float()
                            feed_flat = complement_feed.reshape_as(w_patches).float()
                            v11_complement_rowspace_errors.append(
                                float(
                                    torch.einsum(
                                        "brs,bsp->brp", basis, complement_flat
                                    )
                                    .abs()
                                    .amax()
                                    .item()
                                )
                            )
                            v11_complement_feed_bfloat16.append(
                                float(
                                    v11_encoder.last_complement_feed_dtype
                                    == torch.bfloat16
                                )
                            )
                            complement_rms = complement_flat.square().mean().sqrt()
                            v11_complement_feed_cast_errors.append(
                                float(
                                    (
                                        (feed_flat - complement_flat)
                                        .square()
                                        .mean()
                                        .sqrt()
                                        / complement_rms.clamp_min(1.0e-24)
                                    ).item()
                                )
                            )
                            feed_row = torch.einsum(
                                "brs,bsp->brp", basis, feed_flat
                            )
                            v11_complement_feed_rowspace_relative.append(
                                float(
                                    (
                                        feed_row.square().mean().sqrt()
                                        / feed_flat.square().mean().sqrt().clamp_min(1.0e-24)
                                    ).item()
                                )
                            )

                        def _matrix_to_patches(matrix: torch.Tensor) -> torch.Tensor:
                            padded = torch.zeros_like(W_pad)
                            padded[:, : int(matrix.shape[1]), :] = matrix.float()
                            return padded.transpose(1, 2).reshape(
                                int(matrix.shape[0]), int(matrix.shape[2]) * int(debug["T"]), p
                            )

                        full_patches = _matrix_to_patches(W_hat)
                        floor_patches = _matrix_to_patches(floor_hat)
                        residual_patches = full_patches - floor_patches
                        floor_sumsq = floor_patches.square().sum()
                        residual_sumsq = residual_patches.square().sum()
                        v10_floor_sumsq += float(floor_sumsq.item())
                        v10_residual_sumsq += float(residual_sumsq.item())
                        per_sample_ratio = (
                            residual_patches.square().mean(dim=(-1, -2)).sqrt()
                            / floor_patches.square().mean(dim=(-1, -2)).sqrt().clamp_min(1.0e-24)
                        )
                        v10_residual_to_floor_rms_max = max(
                            v10_residual_to_floor_rms_max,
                            float(per_sample_ratio.amax().item()),
                        )
                        residual_row = torch.einsum("brs,bsp->brp", basis, residual_patches)
                        v10_residual_rowspace_errors.append(
                            float(residual_row.abs().amax().item())
                        )
                        v10_floor_residual_inner_products.append(
                            float(
                                (floor_patches * residual_patches)
                                .sum(dim=(-1, -2))
                                .abs()
                                .amax()
                                .item()
                            )
                        )
                        dot = (floor_patches * residual_patches).sum(dim=(-1, -2))
                        denom = (
                            floor_patches.square().sum(dim=(-1, -2)).sqrt()
                            * residual_patches.square().sum(dim=(-1, -2)).sqrt()
                        ).clamp_min(1.0e-24)
                        v10_floor_residual_cosines.append(
                            float((dot / denom).abs().amax().item())
                        )
                    if v11_enabled:
                        manual_hat, *_rest = raw_model._decode_from_decoder_latent(
                            debug["latent_decoder_z"],
                            dist_patch_by_patch=debug["dist_patch_by_patch"],
                            patch_mask=debug["patch_mask"],
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                            d_in=int(W.shape[1]),
                            d_out=int(W.shape[2]),
                            d_in_pad=int(debug["d_in_pad"]),
                            T=int(debug["T"]),
                            disable_z_shortcut=True,
                        )
                        difference = manual_hat.float() - W_hat.float()
                        v11_manual_decode_errors.append(
                            float(difference.abs().amax().item())
                        )
                        v11_manual_decode_relative_errors.append(
                            float(
                                (
                                    difference.square().mean().sqrt()
                                    / W_hat.float().square().mean().sqrt().clamp_min(1.0e-24)
                                ).item()
                            )
                        )
                if refiner is not None and refiner.last_carrier_state is not None:
                    carrier_chunks.append(refiner.last_carrier_state.cpu())
                if refiner is not None and refiner.last_adaptive_state is not None:
                    adaptive_chunks.append(refiner.last_adaptive_state.cpu())
                replay.append(
                    (
                        debug["dist_patch_by_patch"].detach().cpu(),
                        debug["patch_mask"].detach().cpu(),
                        d_in_cpu,
                        d_out_cpu,
                        torch.tensor(int(debug["T"])),
                        torch.tensor(int(debug["d_in_pad"])),
                    )
                )
                if refiner is not None and not refiner.bypass_refinement:
                    if stage_chunks is None:
                        stage_chunks = [[] for _ in refiner.last_stage_states]
                    if len(stage_chunks) != len(refiner.last_stage_states):
                        raise RuntimeError("V9 stage telemetry depth changed during evaluation")
                    for stage_index, state in enumerate(refiner.last_stage_states):
                        stage_chunks[stage_index].append(state.cpu())
                    chunk_routing: list[dict[str, float]] = []
                    for block in refiner.blocks:
                        diagnostics = block.last_routing_diagnostics
                        if diagnostics is None:
                            raise RuntimeError("V9 routing diagnostics were not captured")
                        chunk_routing.append(
                            {
                                name: float(value.item())
                                for name, value in diagnostics.items()
                                if name != "anchor_family_id"
                            }
                        )
                    routing_rows.append(chunk_routing)
                    refiner.capture_stage_states = False
                    for block in refiner.blocks:
                        block.capture_routing_diagnostics = False
                zero_w_hat, *_unused, zero_w_debug = raw_model.forward_debug(
                    torch.zeros_like(W),
                    x,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    disable_z_shortcut=True,
                )
                zero_w_chunks.append(zero_w_hat.detach().cpu())
                zero_w_latent_chunks.append(
                    zero_w_debug["latent_decoder_z"].detach().cpu()
                )
                zero_z_hat, *_rest = raw_model._decode_from_decoder_latent(
                    torch.zeros_like(debug["latent_decoder_z"]),
                    dist_patch_by_patch=debug["dist_patch_by_patch"],
                    patch_mask=debug["patch_mask"],
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    d_in=int(W.shape[1]),
                    d_out=int(W.shape[2]),
                    d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]),
                    disable_z_shortcut=True,
                )
                zero_z_chunks.append(zero_z_hat.detach().cpu())

            matched = torch.cat(matched_chunks, dim=0)
            latents = torch.cat(latent_chunks, dim=0)
            zero_w_output = torch.cat(zero_w_chunks, dim=0)
            zero_w_latents = torch.cat(zero_w_latent_chunks, dim=0)
            zero_z_output = torch.cat(zero_z_chunks, dim=0)
            targets = target_cpu
            v10_expert_codes_all: torch.Tensor | None = None
            v10_family_leaveout_outputs: dict[str, torch.Tensor] = {}
            v11_trunk_codes_all: torch.Tensor | None = None
            v11_trunk_leaveout_outputs: dict[int, torch.Tensor] = {}
            if v10_enabled:
                if v10_expert_code_chunks is None:
                    raise RuntimeError("V10 matched expert-code telemetry is incomplete")
                v10_expert_codes_all = torch.stack(
                    [torch.cat(chunks, dim=0) for chunks in v10_expert_code_chunks],
                    dim=0,
                )
                if int(step) == 1984:
                    protected_dim = int(raw_model.cfg.big_vae.v10_protected_dim)
                    family_names = ("local", "medium", "broad", "global")
                    for family_index, family_name in enumerate(family_names):
                        leaveout_z = _v10_family_leaveout_latent(
                            latents,
                            v10_expert_codes_all,
                            protected_dim=protected_dim,
                            family_index=family_index,
                        )
                        leaveout_chunks: list[torch.Tensor] = []
                        offset = 0
                        for chunk_index, (W_cpu, _x, _xm, d_in_cpu, d_out_cpu, indices) in enumerate(
                            microbatches
                        ):
                            size = len(indices)
                            replay_row = replay[chunk_index]
                            leaveout_hat, *_rest = raw_model._decode_from_decoder_latent(
                                leaveout_z[offset : offset + size].to(device),
                                dist_patch_by_patch=replay_row[0].to(device),
                                patch_mask=replay_row[1].to(device),
                                d_in_mask=d_in_cpu.to(device),
                                d_out_mask=d_out_cpu.to(device),
                                d_in=int(W_cpu.shape[1]),
                                d_out=int(W_cpu.shape[2]),
                                d_in_pad=int(replay_row[5].item()),
                                T=int(replay_row[4].item()),
                                disable_z_shortcut=True,
                            )
                            leaveout_chunks.append(leaveout_hat.detach().cpu())
                            offset += size
                        if offset != int(latents.shape[0]):
                            raise RuntimeError("V10 family leave-out replay lost samples")
                        v10_family_leaveout_outputs[family_name] = torch.cat(
                            leaveout_chunks, dim=0
                        )
            if v11_enabled:
                if v11_trunk_code_chunks is None:
                    raise RuntimeError("V11 matched trunk-code telemetry is incomplete")
                v11_trunk_codes_all = torch.stack(
                    [torch.cat(chunks, dim=0) for chunks in v11_trunk_code_chunks],
                    dim=0,
                )
                if int(step) == 1984:
                    protected_dim = int(raw_model.cfg.big_vae.v10_protected_dim)
                    trunk_dim = int(raw_model.cfg.big_vae.v11_trunk_dim)
                    for trunk_index in range(int(raw_model.cfg.big_vae.v11_num_trunks)):
                        leaveout_z = latents.view(
                            int(latents.shape[0]),
                            int(raw_model.cfg.big_vae.num_latents),
                            int(raw_model.cfg.big_vae.d_lat),
                        ).clone()
                        start = protected_dim + trunk_index * trunk_dim
                        leaveout_z[..., start : start + trunk_dim] = 0
                        leaveout_chunks = []
                        offset = 0
                        for chunk_index, (W_cpu, _x, _xm, d_in_cpu, d_out_cpu, indices) in enumerate(
                            microbatches
                        ):
                            size = len(indices)
                            replay_row = replay[chunk_index]
                            leaveout_hat, *_rest = raw_model._decode_from_decoder_latent(
                                leaveout_z[offset : offset + size].to(device),
                                dist_patch_by_patch=replay_row[0].to(device),
                                patch_mask=replay_row[1].to(device),
                                d_in_mask=d_in_cpu.to(device),
                                d_out_mask=d_out_cpu.to(device),
                                d_in=int(W_cpu.shape[1]),
                                d_out=int(W_cpu.shape[2]),
                                d_in_pad=int(replay_row[5].item()),
                                T=int(replay_row[4].item()),
                                disable_z_shortcut=True,
                            )
                            leaveout_chunks.append(leaveout_hat.detach().cpu())
                            offset += size
                        if offset != int(latents.shape[0]):
                            raise RuntimeError("V11 trunk leave-out replay lost samples")
                        v11_trunk_leaveout_outputs[trunk_index] = torch.cat(
                            leaveout_chunks, dim=0
                        )

            round_indices = [62] if int(step) != 1984 else [59, 60, 61, 62]
            derangement_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            v10_component_derangements: tuple[torch.Tensor, torch.Tensor] | None = None
            v9a_carrier_only_chunks: list[torch.Tensor] = []
            v9a_adaptive_shuffle_chunks: list[torch.Tensor] = []
            v9a_enabled = bool(
                refiner is not None and getattr(refiner, "carrier_mean_content", False)
            )
            if v9a_enabled:
                if len(carrier_chunks) != len(microbatches) or len(adaptive_chunks) != len(
                    microbatches
                ):
                    raise RuntimeError("V9-A evaluation did not capture every carrier/adaptive chunk")
                carrier_all = torch.cat(carrier_chunks, dim=0)
                adaptive_all = torch.cat(adaptive_chunks, dim=0)
            else:
                carrier_all = None
                adaptive_all = None
            for round_index in round_indices:
                partner: dict[tuple[str, str], tuple[str, str]] = {}
                for left, right in self.source._rounds[round_index]:
                    partner[left] = right
                    partner[right] = left
                shuffled_indices = [
                    lookup[(partner[key], local_index)] for key, local_index in assignments
                ]
                shuffle_index_cpu = torch.tensor(shuffled_indices, dtype=torch.long)
                z_shuffle_chunks: list[torch.Tensor] = []
                w_shuffle_chunks: list[torch.Tensor] = []
                x_shuffle_chunks: list[torch.Tensor] = []
                v10_zc_shuffle_chunks: list[torch.Tensor] = []
                v10_za_shuffle_chunks: list[torch.Tensor] = []
                offset = 0
                for chunk_index, (_W, _x, _xm, _di, _do, indices) in enumerate(microbatches):
                    size = len(indices)
                    chunk_indices = shuffle_index_cpu[offset : offset + size]
                    W = target_cpu[offset : offset + size].to(device)
                    x = x_cpu[offset : offset + size].to(device)
                    x_mask = x_mask_cpu[offset : offset + size].to(device)
                    d_in_mask = d_in_mask_cpu[offset : offset + size].to(device)
                    d_out_mask = d_out_mask_cpu[offset : offset + size].to(device)
                    replay_row = replay[chunk_index]
                    dist_patch, patch_mask = replay_row[0].to(device), replay_row[1].to(device)
                    z_hat, *_rest = raw_model._decode_from_decoder_latent(
                        latents.index_select(0, chunk_indices).to(device),
                        dist_patch_by_patch=dist_patch,
                        patch_mask=patch_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                        d_in=int(W.shape[1]),
                        d_out=int(W.shape[2]),
                        d_in_pad=int(replay_row[5].item()),
                        T=int(replay_row[4].item()),
                        disable_z_shortcut=True,
                    )
                    if round_index == 62 and orthogonal_enabled:
                        protected_dim = int(raw_model.cfg.big_vae.v10_protected_dim)
                        matched_z = latents[offset : offset + size].to(device).view(
                            size,
                            int(raw_model.cfg.big_vae.num_latents),
                            int(raw_model.cfg.big_vae.d_lat),
                        )
                        shuffled_z = latents.index_select(0, chunk_indices).to(device).view_as(
                            matched_z
                        )
                        zc_shuffle = matched_z.clone()
                        zc_shuffle[..., :protected_dim] = shuffled_z[..., :protected_dim]
                        za_shuffle = matched_z.clone()
                        za_shuffle[..., protected_dim:] = shuffled_z[..., protected_dim:]
                        zc_hat, *_rest = raw_model._decode_from_decoder_latent(
                            zc_shuffle,
                            dist_patch_by_patch=dist_patch,
                            patch_mask=patch_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                            d_in=int(W.shape[1]),
                            d_out=int(W.shape[2]),
                            d_in_pad=int(replay_row[5].item()),
                            T=int(replay_row[4].item()),
                            disable_z_shortcut=True,
                        )
                        za_hat, *_rest = raw_model._decode_from_decoder_latent(
                            za_shuffle,
                            dist_patch_by_patch=dist_patch,
                            patch_mask=patch_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                            d_in=int(W.shape[1]),
                            d_out=int(W.shape[2]),
                            d_in_pad=int(replay_row[5].item()),
                            T=int(replay_row[4].item()),
                            disable_z_shortcut=True,
                        )
                        v10_zc_shuffle_chunks.append(zc_hat.detach().cpu())
                        v10_za_shuffle_chunks.append(za_hat.detach().cpu())
                    if round_index == 62 and v9a_enabled:
                        assert carrier_all is not None and adaptive_all is not None
                        carrier_only_z = (
                            (1.0 - float(refiner.carrier_mix))
                            * carrier_all[offset : offset + size]
                        ).to(device)
                        adaptive_shuffle_z = (
                            carrier_only_z
                            + adaptive_all.index_select(0, chunk_indices).to(device)
                        )
                        carrier_only_hat, *_rest = raw_model._decode_from_decoder_latent(
                            carrier_only_z,
                            dist_patch_by_patch=dist_patch,
                            patch_mask=patch_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                            d_in=int(W.shape[1]),
                            d_out=int(W.shape[2]),
                            d_in_pad=int(replay_row[5].item()),
                            T=int(replay_row[4].item()),
                            disable_z_shortcut=True,
                        )
                        adaptive_shuffle_hat, *_rest = raw_model._decode_from_decoder_latent(
                            adaptive_shuffle_z,
                            dist_patch_by_patch=dist_patch,
                            patch_mask=patch_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                            d_in=int(W.shape[1]),
                            d_out=int(W.shape[2]),
                            d_in_pad=int(replay_row[5].item()),
                            T=int(replay_row[4].item()),
                            disable_z_shortcut=True,
                        )
                        v9a_carrier_only_chunks.append(carrier_only_hat.detach().cpu())
                        v9a_adaptive_shuffle_chunks.append(
                            adaptive_shuffle_hat.detach().cpu()
                        )
                    w_hat, *_rest, _debug = raw_model.forward_debug(
                        target_cpu.index_select(0, chunk_indices).to(device),
                        x,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                        disable_z_shortcut=True,
                    )
                    if round_index == 62 and v10_enabled:
                        assert v10_encoder is not None
                        if len(v10_encoder.last_expert_codes) != len(v10_encoder.experts):
                            raise RuntimeError("V10 fixed-X W-shuffle did not capture every expert")
                        if v10_fixed_x_w_expert_code_chunks is None:
                            v10_fixed_x_w_expert_code_chunks = [
                                [] for _ in v10_encoder.last_expert_codes
                            ]
                        for expert_index, code in enumerate(v10_encoder.last_expert_codes):
                            v10_fixed_x_w_expert_code_chunks[expert_index].append(code.cpu())
                    if round_index == 62 and v11_enabled:
                        assert v11_encoder is not None
                        if len(v11_encoder.last_trunk_codes) != len(v11_encoder.trunks):
                            raise RuntimeError("V11 fixed-X W-shuffle did not capture every trunk")
                        if v11_fixed_x_w_trunk_code_chunks is None:
                            v11_fixed_x_w_trunk_code_chunks = [
                                [] for _ in v11_encoder.last_trunk_codes
                            ]
                        for trunk_index, code in enumerate(v11_encoder.last_trunk_codes):
                            v11_fixed_x_w_trunk_code_chunks[trunk_index].append(code.cpu())
                    x_hat, *_rest, _debug = raw_model.forward_debug(
                        W,
                        x_cpu.index_select(0, chunk_indices).to(device),
                        x_mask=x_mask_cpu.index_select(0, chunk_indices).to(device),
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                        disable_z_shortcut=True,
                    )
                    z_shuffle_chunks.append(z_hat.detach().cpu())
                    w_shuffle_chunks.append(w_hat.detach().cpu())
                    x_shuffle_chunks.append(x_hat.detach().cpu())
                    offset += size
                derangement_outputs[round_index] = (
                    torch.cat(z_shuffle_chunks),
                    torch.cat(w_shuffle_chunks),
                    torch.cat(x_shuffle_chunks),
                )
                if round_index == 62 and orthogonal_enabled:
                    v10_component_derangements = (
                        torch.cat(v10_zc_shuffle_chunks),
                        torch.cat(v10_za_shuffle_chunks),
                    )

            metrics: dict[str, float] = {
                "step": float(step),
                "operator_count": float(len(self.source._keys)),
                "tile_count": float(len(assignments)),
                "reserved_derangement_round": 62.0,
            }
            if orthogonal_enabled:
                if not v10_floor_chunks or v10_component_derangements is None:
                    raise RuntimeError("V10 evaluation failed to materialize floor/component controls")
                metrics.update(
                    {
                        "v10_backend_cuda_matmul_allow_tf32": float(
                            bool(torch.backends.cuda.matmul.allow_tf32)
                        ),
                        "v10_backend_cudnn_allow_tf32": float(
                            bool(torch.backends.cudnn.allow_tf32)
                        ),
                        "v10_backend_float32_matmul_precision_highest": float(
                            torch.get_float32_matmul_precision() == "highest"
                        ),
                        "v10_basis_orthogonality_max_abs": max(v10_orthogonality_errors),
                        "v10_basis_min_valid_column_coverage": min(
                            v10_basis_min_valid_column_coverages
                        ),
                        "v10_zc_analysis_max_abs": max(v10_analysis_errors),
                        "v10_residual_rowspace_max_abs": max(v10_residual_rowspace_errors),
                        "v10_floor_residual_inner_product_max_abs": max(
                            v10_floor_residual_inner_products
                        ),
                        "v10_floor_residual_cosine_max_abs": max(
                            v10_floor_residual_cosines
                        ),
                        "v10_residual_to_floor_rms": math.sqrt(
                            v10_residual_sumsq / max(1.0e-24, v10_floor_sumsq)
                        ),
                        "v10_residual_to_floor_rms_max": v10_residual_to_floor_rms_max,
                        "v10_zero_w_latent_max_abs": float(
                            zero_w_latents.float().abs().amax().item()
                        ),
                        "v10_zero_w_output_max_abs": float(
                            zero_w_output.float().abs().amax().item()
                        ),
                        "v10_zero_z_output_max_abs": float(
                            zero_z_output.float().abs().amax().item()
                        ),
                    }
                )
                if v11_enabled:
                    if (
                        not v11_complement_rowspace_errors
                        or not v11_complement_feed_bfloat16
                    ):
                        raise RuntimeError("V11 complement input telemetry is incomplete")
                    metrics.update(
                        {
                            "v11_complement_input_rowspace_max_abs": max(
                                v11_complement_rowspace_errors
                            ),
                            "v11_complement_feed_is_bfloat16": min(
                                v11_complement_feed_bfloat16
                            ),
                            "v11_complement_feed_cast_relative_rms": max(
                                v11_complement_feed_cast_errors
                            ),
                            "v11_complement_feed_rowspace_relative_rms": max(
                                v11_complement_feed_rowspace_relative
                            ),
                            "v11_native_latent_capture_max_abs": max(
                                v11_latent_capture_errors
                            ),
                            "v11_manual_decode_max_abs": max(v11_manual_decode_errors),
                            "v11_manual_decode_relative_rms": max(
                                v11_manual_decode_relative_errors
                            ),
                        }
                    )
            matched_dirs: list[float] = []
            matched_scales: list[float] = []
            matched_nrmse: list[float] = []
            z_shuffle_dir_delta: list[float] = []
            w_shuffle_dir_delta: list[float] = []
            x_shuffle_dir_delta: list[float] = []
            adaptive_shuffle_dir_delta: list[float] = []
            oracle_dirs: list[float] = []
            matched_minus_oracle_dirs: list[float] = []
            floor_dirs: list[float] = []
            residual_to_floor_rms: list[float] = []
            zc_shuffle_dir_delta: list[float] = []
            za_shuffle_dir_delta: list[float] = []
            v10_family_leaveout_dir_deltas: dict[str, list[float]] = {
                family: [] for family in v10_family_leaveout_outputs
            }
            v11_trunk_leaveout_dir_deltas: dict[int, list[float]] = {
                trunk: [] for trunk in v11_trunk_leaveout_outputs
            }
            v11_complement_nrmse: list[float] = []
            reserved_z, reserved_w, reserved_x = derangement_outputs[62]
            v10_floor_output = (
                torch.cat(v10_floor_chunks, dim=0) if orthogonal_enabled else None
            )
            if v10_component_derangements is not None:
                v10_zc_shuffle_output, v10_za_shuffle_output = v10_component_derangements
            else:
                v10_zc_shuffle_output = v10_za_shuffle_output = None
            for name, shuffled_output in (
                ("z_shuffle_output_relative_delta", reserved_z),
                ("fixed_x_w_shuffle_output_relative_delta", reserved_w),
            ):
                output_pair_rms = (
                    0.5
                    * (
                        matched.float().square().mean()
                        + shuffled_output.float().square().mean()
                    )
                ).sqrt()
                metrics[name] = float(
                    (
                        (matched.float() - shuffled_output.float()).square().mean().sqrt()
                        / output_pair_rms.clamp_min(1.0e-24)
                    ).item()
                )
            if v9a_enabled:
                carrier_only_output = torch.cat(v9a_carrier_only_chunks, dim=0)
                adaptive_shuffle_output = torch.cat(
                    v9a_adaptive_shuffle_chunks, dim=0
                )
                for name, alternate in (
                    ("v9a_full_vs_carrier_only_output_relative_delta", carrier_only_output),
                    ("v9a_adaptive_shuffle_output_relative_delta", adaptive_shuffle_output),
                ):
                    pair_rms = (
                        0.5
                        * (
                            matched.float().square().mean()
                            + alternate.float().square().mean()
                        )
                    ).sqrt()
                    metrics[name] = float(
                        (
                            (matched.float() - alternate.float()).square().mean().sqrt()
                            / pair_rms.clamp_min(1.0e-24)
                        ).item()
                    )
                metrics["v9a_zero_w_latent_max_abs"] = float(
                    zero_w_latents.float().abs().max().item()
                )
            target_matrices = torch.stack(
                [_stitch(targets, assignments, self.source, key) for key in self.source._keys]
            ).float()
            oracle_matrix = _weighted_cosine_position_oracle(
                target_matrices,
                patch_size=self.patch_size,
                gamma=self.gamma,
            )
            for operator_index, key in enumerate(self.source._keys):
                target_matrix = target_matrices[operator_index]
                matched_matrix = _stitch(matched, assignments, self.source, key)
                z_shuffle_matrix = _stitch(reserved_z, assignments, self.source, key)
                w_shuffle_matrix = _stitch(reserved_w, assignments, self.source, key)
                x_shuffle_matrix = _stitch(reserved_x, assignments, self.source, key)
                zero_w_matrix = _stitch(zero_w_output, assignments, self.source, key)
                zero_z_matrix = _stitch(zero_z_output, assignments, self.source, key)
                floor_matrix = (
                    _stitch(v10_floor_output, assignments, self.source, key)
                    if v10_floor_output is not None
                    else None
                )
                zc_shuffle_matrix = (
                    _stitch(v10_zc_shuffle_output, assignments, self.source, key)
                    if v10_zc_shuffle_output is not None
                    else None
                )
                za_shuffle_matrix = (
                    _stitch(v10_za_shuffle_output, assignments, self.source, key)
                    if v10_za_shuffle_output is not None
                    else None
                )
                family_leaveout_matrices = {
                    family: _stitch(output, assignments, self.source, key)
                    for family, output in v10_family_leaveout_outputs.items()
                }
                trunk_leaveout_matrices = {
                    trunk: _stitch(output, assignments, self.source, key)
                    for trunk, output in v11_trunk_leaveout_outputs.items()
                }
                adaptive_shuffle_matrix = (
                    _stitch(adaptive_shuffle_output, assignments, self.source, key)
                    if v9a_enabled
                    else None
                )
                matched_metrics = _loss_metrics(
                    target_matrix,
                    matched_matrix,
                    patch_size=self.patch_size,
                    gamma=self.gamma,
                    lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale,
                    huber_delta=self.huber_delta,
                )
                z_metrics = _loss_metrics(
                    target_matrix,
                    z_shuffle_matrix,
                    patch_size=self.patch_size,
                    gamma=self.gamma,
                    lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale,
                    huber_delta=self.huber_delta,
                )
                w_metrics = _loss_metrics(
                    target_matrix,
                    w_shuffle_matrix,
                    patch_size=self.patch_size,
                    gamma=self.gamma,
                    lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale,
                    huber_delta=self.huber_delta,
                )
                x_metrics = _loss_metrics(
                    target_matrix, x_shuffle_matrix, patch_size=self.patch_size,
                    gamma=self.gamma, lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                )
                zero_w_metrics = _loss_metrics(
                    target_matrix, zero_w_matrix, patch_size=self.patch_size,
                    gamma=self.gamma, lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                )
                zero_z_metrics = _loss_metrics(
                    target_matrix, zero_z_matrix, patch_size=self.patch_size,
                    gamma=self.gamma, lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                )
                oracle_metrics = _loss_metrics(
                    target_matrix, oracle_matrix, patch_size=self.patch_size,
                    gamma=self.gamma, lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                )
                prefix = f"operator_{operator_index:02d}"
                metrics[f"{prefix}_checkpoint_sha256"] = key[0]
                metrics[f"{prefix}_matched_total"] = matched_metrics["total"]
                metrics[f"{prefix}_matched_dir"] = matched_metrics["dir"]
                metrics[f"{prefix}_matched_scale"] = matched_metrics["scale"]
                metrics[f"{prefix}_matched_nrmse"] = _nrmse(target_matrix, matched_matrix)
                metrics[f"{prefix}_z_shuffle_dir_delta"] = z_metrics["dir"] - matched_metrics["dir"]
                metrics[f"{prefix}_fixed_x_w_shuffle_dir_delta"] = (
                    w_metrics["dir"] - matched_metrics["dir"]
                )
                metrics[f"{prefix}_fixed_w_x_shuffle_dir_delta"] = (
                    x_metrics["dir"] - matched_metrics["dir"]
                )
                metrics[f"{prefix}_zero_w_dir"] = zero_w_metrics["dir"]
                metrics[f"{prefix}_zero_w_total"] = zero_w_metrics["total"]
                metrics[f"{prefix}_zero_z_dir"] = zero_z_metrics["dir"]
                metrics[f"{prefix}_zero_z_total"] = zero_z_metrics["total"]
                metrics[f"{prefix}_weighted_cosine_position_oracle_dir"] = oracle_metrics["dir"]
                if floor_matrix is not None and zc_shuffle_matrix is not None and za_shuffle_matrix is not None:
                    floor_metrics = _loss_metrics(
                        target_matrix, floor_matrix, patch_size=self.patch_size,
                        gamma=self.gamma, lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                    )
                    zc_metrics = _loss_metrics(
                        target_matrix, zc_shuffle_matrix, patch_size=self.patch_size,
                        gamma=self.gamma, lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                    )
                    za_metrics = _loss_metrics(
                        target_matrix, za_shuffle_matrix, patch_size=self.patch_size,
                        gamma=self.gamma, lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                    )
                    metrics[f"{prefix}_v10_floor_dir"] = floor_metrics["dir"]
                    operator_residual_to_floor = float(
                        (
                            (matched_matrix.float() - floor_matrix.float()).square().mean().sqrt()
                            / floor_matrix.float().square().mean().sqrt().clamp_min(1.0e-24)
                        ).item()
                    )
                    metrics[f"{prefix}_v10_residual_to_floor_rms"] = operator_residual_to_floor
                    metrics[f"{prefix}_v10_zc_shuffle_dir_delta"] = (
                        zc_metrics["dir"] - matched_metrics["dir"]
                    )
                    metrics[f"{prefix}_v10_za_shuffle_dir_delta"] = (
                        za_metrics["dir"] - matched_metrics["dir"]
                    )
                    floor_dirs.append(floor_metrics["dir"])
                    residual_to_floor_rms.append(operator_residual_to_floor)
                    zc_shuffle_dir_delta.append(zc_metrics["dir"] - matched_metrics["dir"])
                    za_shuffle_dir_delta.append(za_metrics["dir"] - matched_metrics["dir"])
                    if v11_enabled:
                        target_complement = target_matrix.float() - floor_matrix.float()
                        predicted_complement = matched_matrix.float() - floor_matrix.float()
                        complement_ratio = float(
                            (
                                (predicted_complement - target_complement)
                                .square()
                                .mean()
                                .sqrt()
                                / target_complement.square().mean().sqrt().clamp_min(1.0e-24)
                            ).item()
                        )
                        metrics[f"{prefix}_v11_complement_nrmse"] = complement_ratio
                        v11_complement_nrmse.append(complement_ratio)
                    for family, leaveout_matrix in family_leaveout_matrices.items():
                        leaveout_metrics = _loss_metrics(
                            target_matrix,
                            leaveout_matrix,
                            patch_size=self.patch_size,
                            gamma=self.gamma,
                            lambda_dir=self.lambda_dir,
                            lambda_scale=self.lambda_scale,
                            huber_delta=self.huber_delta,
                        )
                        delta = leaveout_metrics["dir"] - matched_metrics["dir"]
                        metrics[f"{prefix}_v10_leaveout_{family}_dir_delta"] = delta
                        v10_family_leaveout_dir_deltas[family].append(delta)
                    for trunk, leaveout_matrix in trunk_leaveout_matrices.items():
                        leaveout_metrics = _loss_metrics(
                            target_matrix,
                            leaveout_matrix,
                            patch_size=self.patch_size,
                            gamma=self.gamma,
                            lambda_dir=self.lambda_dir,
                            lambda_scale=self.lambda_scale,
                            huber_delta=self.huber_delta,
                        )
                        delta = leaveout_metrics["dir"] - matched_metrics["dir"]
                        metrics[f"{prefix}_v11_trunk_{trunk}_leaveout_dir_delta"] = delta
                        v11_trunk_leaveout_dir_deltas[trunk].append(delta)
                if adaptive_shuffle_matrix is not None:
                    adaptive_shuffle_metrics = _loss_metrics(
                        target_matrix,
                        adaptive_shuffle_matrix,
                        patch_size=self.patch_size,
                        gamma=self.gamma,
                        lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale,
                        huber_delta=self.huber_delta,
                    )
                    delta = adaptive_shuffle_metrics["dir"] - matched_metrics["dir"]
                    metrics[f"{prefix}_v9a_adaptive_shuffle_dir_delta"] = delta
                    adaptive_shuffle_dir_delta.append(delta)
                matched_dirs.append(matched_metrics["dir"])
                matched_scales.append(matched_metrics["scale"])
                matched_nrmse.append(metrics[f"{prefix}_matched_nrmse"])
                z_shuffle_dir_delta.append(metrics[f"{prefix}_z_shuffle_dir_delta"])
                w_shuffle_dir_delta.append(metrics[f"{prefix}_fixed_x_w_shuffle_dir_delta"])
                x_shuffle_dir_delta.append(metrics[f"{prefix}_fixed_w_x_shuffle_dir_delta"])
                oracle_dirs.append(oracle_metrics["dir"])
                matched_minus_oracle_dirs.append(matched_metrics["dir"] - oracle_metrics["dir"])
            metrics.update(
                {
                    "matched_mean_dir": sum(matched_dirs) / len(matched_dirs),
                    "matched_max_dir": max(matched_dirs),
                    "matched_max_scale": max(matched_scales),
                    "matched_max_nrmse": max(matched_nrmse),
                    "z_shuffle_min_dir_delta": min(z_shuffle_dir_delta),
                    "fixed_x_w_shuffle_min_dir_delta": min(w_shuffle_dir_delta),
                    "fixed_w_x_shuffle_min_dir_delta": min(x_shuffle_dir_delta),
                    "weighted_cosine_position_oracle_mean_dir": sum(oracle_dirs) / len(oracle_dirs),
                    "model_beats_weighted_oracle_by_0p05_fraction": sum(
                        value <= -0.05 for value in matched_minus_oracle_dirs
                    ) / len(matched_minus_oracle_dirs),
                    "z_shuffle_positive_fraction": sum(value > 0.0 for value in z_shuffle_dir_delta) / len(z_shuffle_dir_delta),
                    "fixed_x_w_shuffle_positive_fraction": sum(value > 0.0 for value in w_shuffle_dir_delta) / len(w_shuffle_dir_delta),
                }
            )
            if floor_dirs:
                floor_tensor = torch.tensor(floor_dirs, dtype=torch.float64)
                metrics.update(
                    {
                        "v10_floor_mean_dir": float(floor_tensor.mean().item()),
                        "v10_floor_p95_dir": float(torch.quantile(floor_tensor, 0.95).item()),
                        "v10_residual_to_floor_rms_mean": float(
                            torch.tensor(residual_to_floor_rms, dtype=torch.float64).mean().item()
                        ),
                        "v10_residual_to_floor_rms_p95": float(
                            torch.quantile(
                                torch.tensor(residual_to_floor_rms, dtype=torch.float64),
                                0.95,
                            ).item()
                        ),
                        "v10_residual_to_floor_rms_operator_max": max(
                            residual_to_floor_rms
                        ),
                        "v10_matched_minus_floor_mean_dir": (
                            sum(matched_dirs) / len(matched_dirs)
                            - float(floor_tensor.mean().item())
                        ),
                        "v10_matched_minus_floor_p95_dir": (
                            float(torch.quantile(torch.tensor(matched_dirs, dtype=torch.float64), 0.95).item())
                            - float(torch.quantile(floor_tensor, 0.95).item())
                        ),
                        "v10_zc_shuffle_min_dir_delta": min(zc_shuffle_dir_delta),
                        "v10_za_shuffle_min_dir_delta": min(za_shuffle_dir_delta),
                        "v10_zc_shuffle_positive_fraction": sum(
                            value > 0 for value in zc_shuffle_dir_delta
                        ) / len(zc_shuffle_dir_delta),
                        "v10_za_shuffle_positive_fraction": sum(
                            value > 0 for value in za_shuffle_dir_delta
                        ) / len(za_shuffle_dir_delta),
                    }
                )
                for name, values in (
                    ("v10_floor_dir", floor_dirs),
                    ("v10_zc_shuffle_dir_delta", zc_shuffle_dir_delta),
                    ("v10_za_shuffle_dir_delta", za_shuffle_dir_delta),
                ):
                    tensor = torch.tensor(values, dtype=torch.float64)
                    for quantile in (0.05, 0.1, 0.5, 0.9, 0.95):
                        metrics[f"{name}_p{int(quantile * 100):02d}"] = float(
                            torch.quantile(tensor, quantile).item()
                        )
            if v10_enabled:
                if (
                    v10_expert_code_chunks is None
                    or v10_fixed_x_w_expert_code_chunks is None
                    or not v10_routing_rows
                ):
                    raise RuntimeError("V10 expert telemetry is incomplete")
                if v10_expert_codes_all is None:
                    raise RuntimeError("V10 matched expert codes were not assembled")
                fixed_x_w_expert_codes = torch.stack(
                    [
                        torch.cat(chunks, dim=0)
                        for chunks in v10_fixed_x_w_expert_code_chunks
                    ],
                    dim=0,
                )
                metrics.update(
                    _v10_expert_code_metrics(
                        v10_expert_codes_all,
                        shuffled_expert_codes=fixed_x_w_expert_codes,
                    )
                )
                for family, output in v10_family_leaveout_outputs.items():
                    pair_rms = (
                        0.5
                        * (
                            matched.float().square().mean()
                            + output.float().square().mean()
                        )
                    ).sqrt()
                    metrics[f"v10_leaveout_{family}_output_relative_delta"] = float(
                        (
                            (matched.float() - output.float()).square().mean().sqrt()
                            / pair_rms.clamp_min(1.0e-24)
                        ).item()
                    )
                    deltas = torch.tensor(
                        v10_family_leaveout_dir_deltas[family], dtype=torch.float64
                    )
                    if int(deltas.numel()) != len(self.source._keys):
                        raise RuntimeError("V10 family leave-out omitted operator metrics")
                    metrics[f"v10_leaveout_{family}_dir_delta_mean"] = float(
                        deltas.mean().item()
                    )
                    metrics[f"v10_leaveout_{family}_dir_delta_p05"] = float(
                        torch.quantile(deltas, 0.05).item()
                    )
                    metrics[f"v10_leaveout_{family}_dir_delta_min"] = float(
                        deltas.min().item()
                    )
                routing_fields = (
                    "entropy_mean",
                    "effective_keys_median",
                    "max_mass_median",
                    "adaptive_score_rms",
                    "adaptive_score_max_abs",
                    "adaptive_score_near_bound_fraction",
                    "unique_argmax_min",
                    "protected_anchor_floor",
                )
                for expert_index in range(len(v10_encoder.experts)):
                    family = ("local", "medium", "broad", "global")[expert_index % 4]
                    for field in routing_fields:
                        values = [row[expert_index][field] for row in v10_routing_rows]
                        aggregate = min(values) if field == "unique_argmax_min" else sum(values) / len(values)
                        metrics[f"v10_expert_{expert_index:02d}_{family}_{field}"] = aggregate
                        if field == "adaptive_score_near_bound_fraction":
                            metrics[
                                f"v10_expert_{expert_index:02d}_{family}_{field}_max"
                            ] = max(values)
            if v11_enabled:
                if (
                    v11_trunk_codes_all is None
                    or v11_trunk_delta_chunks is None
                    or v11_fixed_x_w_trunk_code_chunks is None
                    or not v11_routing_rows
                    or v11_encoder is None
                ):
                    raise RuntimeError("V11 trunk/routing telemetry is incomplete")
                fixed_x_w_trunk_codes = torch.stack(
                    [
                        torch.cat(chunks, dim=0)
                        for chunks in v11_fixed_x_w_trunk_code_chunks
                    ],
                    dim=0,
                )
                metrics.update(
                    _v11_trunk_code_metrics(
                        v11_trunk_codes_all,
                        shuffled_trunk_codes=fixed_x_w_trunk_codes,
                    )
                )
                for trunk_index, chunks in enumerate(v11_trunk_delta_chunks):
                    delta = torch.cat(chunks, dim=0).float()
                    code = v11_trunk_codes_all[trunk_index].float()
                    delta_rms = delta.square().mean().sqrt()
                    metrics[f"v11_trunk_{trunk_index:02d}_delta_rms"] = float(
                        delta_rms.item()
                    )
                    metrics[
                        f"v11_trunk_{trunk_index:02d}_code_to_delta_rms"
                    ] = float(
                        (
                            code.square().mean().sqrt()
                            / delta_rms.clamp_min(1.0e-24)
                        ).item()
                    )
                if v11_complement_nrmse:
                    complement_tensor = torch.tensor(
                        v11_complement_nrmse, dtype=torch.float64
                    )
                    metrics.update(
                        {
                            "v11_complement_nrmse_mean": float(
                                complement_tensor.mean().item()
                            ),
                            "v11_complement_nrmse_p95": float(
                                torch.quantile(complement_tensor, 0.95).item()
                            ),
                            "v11_complement_nrmse_max": float(
                                complement_tensor.max().item()
                            ),
                            "v11_complement_normalized_mse_mean": float(
                                complement_tensor.square().mean().item()
                            ),
                        }
                    )
                for trunk_index, output in v11_trunk_leaveout_outputs.items():
                    pair_rms = (
                        0.5
                        * (
                            matched.float().square().mean()
                            + output.float().square().mean()
                        )
                    ).sqrt()
                    metrics[
                        f"v11_trunk_{trunk_index:02d}_leaveout_output_relative_delta"
                    ] = float(
                        (
                            (matched.float() - output.float()).square().mean().sqrt()
                            / pair_rms.clamp_min(1.0e-24)
                        ).item()
                    )
                    deltas = torch.tensor(
                        v11_trunk_leaveout_dir_deltas[trunk_index],
                        dtype=torch.float64,
                    )
                    if int(deltas.numel()) != len(self.source._keys):
                        raise RuntimeError("V11 trunk leave-out omitted operator metrics")
                    metrics[
                        f"v11_trunk_{trunk_index:02d}_leaveout_dir_delta_mean"
                    ] = float(deltas.mean().item())
                    metrics[
                        f"v11_trunk_{trunk_index:02d}_leaveout_dir_delta_p05"
                    ] = float(torch.quantile(deltas, 0.05).item())
                    metrics[
                        f"v11_trunk_{trunk_index:02d}_leaveout_dir_delta_min"
                    ] = float(deltas.min().item())
                routing_fields = (
                    "entropy_mean",
                    "effective_keys_median",
                    "max_mass_median",
                    "cosine_score_rms",
                    "adaptive_score_rms",
                    "adaptive_score_max_abs",
                    "adaptive_score_near_bound_fraction",
                    "unique_argmax_min",
                    "protected_anchor_floor",
                    "relative_log_rms_feature_rms",
                    "norm_key_contribution_rms",
                    "raw_content_rms",
                    "out_correction_rms",
                    "ffn_correction_rms",
                )
                for block_index, _block in enumerate(v11_encoder.routing_blocks):
                    trunk_index = block_index // int(v11_encoder.blocks_per_trunk)
                    local_index = block_index % int(v11_encoder.blocks_per_trunk)
                    family = ("local", "medium", "broad", "global")[local_index % 4]
                    for field in routing_fields:
                        values = [row[block_index][field] for row in v11_routing_rows]
                        aggregate = (
                            min(values)
                            if field == "unique_argmax_min"
                            else sum(values) / len(values)
                        )
                        prefix = (
                            f"v11_trunk_{trunk_index:02d}_block_{local_index:02d}_"
                            f"{family}_{field}"
                        )
                        metrics[prefix] = aggregate
                        if field == "adaptive_score_near_bound_fraction":
                            metrics[f"{prefix}_max"] = max(values)
            if adaptive_shuffle_dir_delta:
                tensor = torch.tensor(adaptive_shuffle_dir_delta, dtype=torch.float64)
                metrics["v9a_adaptive_shuffle_min_dir_delta"] = min(
                    adaptive_shuffle_dir_delta
                )
                metrics["v9a_adaptive_shuffle_positive_fraction"] = sum(
                    value > 0.0 for value in adaptive_shuffle_dir_delta
                ) / len(adaptive_shuffle_dir_delta)
                for quantile in (0.05, 0.1, 0.5):
                    metrics[
                        f"v9a_adaptive_shuffle_dir_delta_p{int(quantile * 100):02d}"
                    ] = float(torch.quantile(tensor, quantile).item())
            for name, values in (
                ("matched_dir", matched_dirs),
                ("matched_scale", matched_scales),
                ("matched_nrmse", matched_nrmse),
                ("matched_minus_weighted_oracle_dir", matched_minus_oracle_dirs),
                ("z_shuffle_dir_delta", z_shuffle_dir_delta),
                ("fixed_x_w_shuffle_dir_delta", w_shuffle_dir_delta),
            ):
                tensor = torch.tensor(values, dtype=torch.float64)
                for quantile in (0.05, 0.1, 0.5, 0.9, 0.95, 0.99):
                    metrics[f"{name}_p{int(quantile * 100):02d}"] = float(
                        torch.quantile(tensor, quantile).item()
                    )
            metrics["worst8_matched_dir_operator_indices"] = sorted(
                range(len(matched_dirs)), key=lambda index: matched_dirs[index], reverse=True
            )[:8]
            metrics["worst8_matched_dir_checkpoint_sha256"] = [
                self.source._keys[index][0]
                for index in metrics["worst8_matched_dir_operator_indices"]
            ]
            for name, values in (
                ("z_shuffle_dir_delta", z_shuffle_dir_delta),
                ("fixed_x_w_shuffle_dir_delta", w_shuffle_dir_delta),
            ):
                lower_indices = sorted(range(len(values)), key=lambda index: values[index])[:8]
                metrics[f"lowest8_{name}_operator_indices"] = lower_indices
                metrics[f"lowest8_{name}_checkpoint_sha256"] = [
                    self.source._keys[index][0] for index in lower_indices
                ]
            for round_index, (z_output, w_output, _x_output) in derangement_outputs.items():
                if round_index == 62:
                    continue
                per_operator_z: list[float] = []
                per_operator_w: list[float] = []
                for key in self.source._keys:
                    target_matrix = _stitch(targets, assignments, self.source, key)
                    matched_matrix = _stitch(matched, assignments, self.source, key)
                    base = _loss_metrics(
                        target_matrix, matched_matrix, patch_size=self.patch_size,
                        gamma=self.gamma, lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale, huber_delta=self.huber_delta,
                    )["dir"]
                    per_operator_z.append(_loss_metrics(
                        target_matrix, _stitch(z_output, assignments, self.source, key),
                        patch_size=self.patch_size, gamma=self.gamma,
                        lambda_dir=self.lambda_dir, lambda_scale=self.lambda_scale,
                        huber_delta=self.huber_delta,
                    )["dir"] - base)
                    per_operator_w.append(_loss_metrics(
                        target_matrix, _stitch(w_output, assignments, self.source, key),
                        patch_size=self.patch_size, gamma=self.gamma,
                        lambda_dir=self.lambda_dir, lambda_scale=self.lambda_scale,
                        huber_delta=self.huber_delta,
                    )["dir"] - base)
                metrics[f"training_partner_round_{round_index:02d}_z_shuffle_min_dir_delta"] = min(per_operator_z)
                metrics[f"training_partner_round_{round_index:02d}_fixed_x_w_shuffle_min_dir_delta"] = min(per_operator_w)
                for name, values in (
                    ("z_shuffle_dir_delta", per_operator_z),
                    ("fixed_x_w_shuffle_dir_delta", per_operator_w),
                ):
                    tensor = torch.tensor(values, dtype=torch.float64)
                    prefix = f"training_partner_round_{round_index:02d}_{name}"
                    metrics[f"{prefix}_p10"] = float(torch.quantile(tensor, 0.1).item())
                    metrics[f"{prefix}_median"] = float(torch.quantile(tensor, 0.5).item())
                    metrics[f"{prefix}_positive_fraction"] = sum(
                        value > 0.0 for value in values
                    ) / len(values)
            if stage_chunks is not None:
                reserved_partner: dict[tuple[str, str], tuple[str, str]] = {}
                for left, right in self.source.eval_derangement_round:
                    reserved_partner[left], reserved_partner[right] = right, left
                shuffle_index = torch.tensor(
                    [lookup[(reserved_partner[key], local_index)] for key, local_index in assignments],
                    dtype=torch.long,
                )
                carrier = torch.cat(carrier_chunks, dim=0).float()
                paired = carrier.index_select(0, shuffle_index)
                carrier_pair_rms = (0.5 * (carrier.square().mean() + paired.square().mean())).sqrt()
                metrics["v9_carrier_shuffle_relative_delta"] = float(
                    ((carrier - paired).square().mean().sqrt() / carrier_pair_rms.clamp_min(1e-24)).item()
                )
                if adaptive_chunks:
                    adaptive = torch.cat(adaptive_chunks, dim=0).float()
                    metrics["v9a_mixed_adaptive_to_carrier_rms"] = float(
                        (
                            adaptive.square().mean().sqrt()
                            / (
                                (1.0 - float(refiner.carrier_mix))
                                * carrier.square().mean().sqrt()
                            ).clamp_min(1.0e-24)
                        ).item()
                    )
                for stage_index, chunks in enumerate(stage_chunks):
                    state = torch.cat(chunks, dim=0).float()
                    shuffled = state.index_select(0, shuffle_index)
                    pair_rms = (0.5 * (state.square().mean() + shuffled.square().mean())).sqrt()
                    metrics[f"v9_stage_{stage_index:02d}_shuffle_relative_delta"] = float(
                        ((state - shuffled).square().mean().sqrt() / pair_rms.clamp_min(1e-24)).item()
                    )
                    metrics[f"v9_stage_{stage_index:02d}_rms"] = float(state.square().mean().sqrt().item())
                for stage_index in range(len(routing_rows[0])):
                    family = ("local", "medium", "broad", "global")[stage_index % 4]
                    for field in (
                        "cosine_score_rms", "adaptive_score_rms", "adaptive_score_max_abs",
                        "adaptive_score_near_bound_fraction", "unique_argmax_min",
                        "relative_log_rms_feature_rms", "norm_key_contribution_rms",
                        "entropy_mean", "effective_keys_median", "max_mass_median",
                        "state_rms", "delta_rms", "scaled_delta_to_state_rms",
                        "raw_content_rms", "out_correction_rms", "ffn_correction_rms",
                    ):
                        values = [row[stage_index][field] for row in routing_rows]
                        metrics[f"v9_stage_{stage_index:02d}_{family}_{field}"] = (
                            min(values) if field == "unique_argmax_min" else sum(values) / len(values)
                        )
                        if field == "adaptive_score_near_bound_fraction":
                            metrics[
                                f"v9_stage_{stage_index:02d}_{family}_{field}_max"
                            ] = max(values)
            if v11_enabled:
                # Common orthogonal-floor telemetry is computed once with V10
                # local names above, then relabelled to the active architecture.
                renamed: dict[str, Any] = {}
                removed: list[str] = []
                for name, value in metrics.items():
                    if name.startswith("v10_"):
                        renamed["v11_" + name[4:]] = value
                        removed.append(name)
                    elif "_v10_" in name:
                        renamed[name.replace("_v10_", "_v11_")] = value
                        removed.append(name)
                for name in removed:
                    del metrics[name]
                metrics.update(renamed)
                if int(step) == 1984:
                    if self.v11_step0_complement_normalized_mse is None:
                        raise RuntimeError("V11 final evaluator lost its step-0 baseline")
                    final_failures = v11_scientific_contract_failures(
                        metrics,
                        step0_complement_normalized_mse=(
                            self.v11_step0_complement_normalized_mse
                        ),
                        final=True,
                    )
                    metrics["v11_final_scientific_contract_pass"] = float(
                        not final_failures
                    )
                    metrics["v11_final_scientific_contract_failures"] = final_failures
                validate_v11_numeric_telemetry_finite(metrics)
            elif v10_enabled:
                validate_v10_numeric_telemetry_finite(metrics)
        finally:
            if refiner is not None:
                refiner.capture_stage_states = False
                refiner.last_stage_states = ()
                refiner.last_carrier_state = None
                refiner.last_adaptive_state = None
                for block in refiner.blocks:
                    block.capture_routing_diagnostics = False
                    block.last_routing_diagnostics = None
            if v10_encoder is not None:
                _reset_v10_evaluator_capture(v10_encoder)
            if v11_encoder is not None:
                _reset_v11_evaluator_capture(v11_encoder)
            raw_model.train(was_training)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        return metrics


__all__ = [
    "CanonicalOperatorSetTrainingDataset",
    "OperatorSetOverfitEvaluator",
    "_weighted_cosine_position_oracle",
    "validate_v9a_step0_metrics",
    "validate_v9a_step1_gradients",
    "validate_v10_geometry_metrics",
    "validate_v10_numeric_telemetry_finite",
    "validate_v10_step0_metrics",
    "validate_v10_step1_gradients",
    "validate_v11_geometry_metrics",
    "validate_v11_numeric_telemetry_finite",
    "validate_v11_step0_metrics",
    "validate_v11_step1_gradients",
    "v11_scientific_contract_failures",
    "v11_complement_normalized_mse",
    "canonical_operator_set_data_pipeline",
]
