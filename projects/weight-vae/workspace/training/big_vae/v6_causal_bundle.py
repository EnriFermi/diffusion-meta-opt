from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "weightclip_ae_v6_two_operator_causal_bundle_v1"
ARMS = (
    "mixed",
    "alternating_homogeneous",
    "cyclic_singleton",
    "centered_value",
    "zero_common_bias",
    "mandatory_cross_refresh",
    "clean_content_readout",
)
_RETURN_PROJECTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"patch_tokenizer\.blocks\.\d+\.(?:base_fc2|cond_fc2)$",
        r"encoder_layers\.\d+\.local_block\.(?:out_proj|ffn\.fc2)$",
        r"encoder_layers\.\d+\.perceiver_block\.(?:cross_out_proj|self_out_proj|ffn\.3)$",
        r"latent_to_weight_feedback\.out$",
        r"encoder_conditioning_adapters\.\d+\.mix_out$",
        r"decoder_layers\.\d+\.(?:self_out_proj|ffn\.3)$",
    )
)


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "sha256": _tensor_sha256(detached),
        "nonzero_count": int(torch.count_nonzero(detached).item()),
        "rms": float(detached.float().square().mean().sqrt().item()),
        "requires_grad": bool(value.requires_grad),
    }


def _unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while hasattr(current, "module") and isinstance(current.module, nn.Module):
        current = current.module
    return current


def _return_projection_bias_names(model: nn.Module) -> tuple[str, ...]:
    root = _unwrap_model(model)
    names: list[str] = []
    for module_name, module in root.named_modules():
        if not isinstance(module, nn.Linear) or module.bias is None:
            continue
        if any(pattern.fullmatch(module_name) for pattern in _RETURN_PROJECTION_PATTERNS):
            names.append(f"{module_name}.bias")
    return tuple(sorted(names))


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "enabled",
        "schema",
        "arm",
        "ledger_path",
        "zero_all_return_biases",
        "freeze_zeroed_biases",
        "homogeneous_first_operator",
        "centered_value_gain_max",
        "cross_refresh_rms",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown V6 causal-bundle config keys: {unknown}")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    if str(raw.get("schema", "")) != SCHEMA:
        raise ValueError(f"V6 causal-bundle schema must be {SCHEMA!r}")
    arm = str(raw.get("arm", "")).strip()
    if arm not in ARMS:
        raise ValueError(f"V6 causal-bundle arm must be one of {ARMS}, got {arm!r}")
    ledger_path = str(raw.get("ledger_path", "")).strip()
    if not ledger_path:
        raise ValueError("V6 causal-bundle ledger_path must be non-empty")
    zero_all = bool(raw.get("zero_all_return_biases", False))
    if zero_all and arm != "zero_common_bias":
        raise ValueError("zero_all_return_biases is valid only for zero_common_bias")
    freeze_zeroed = bool(raw.get("freeze_zeroed_biases", False))
    if freeze_zeroed and arm != "zero_common_bias":
        raise ValueError("freeze_zeroed_biases is valid only for zero_common_bias")
    cross_refresh_rms = float(raw.get("cross_refresh_rms", 1.0))
    if arm == "mandatory_cross_refresh" and cross_refresh_rms != 1.0:
        raise ValueError("mandatory_cross_refresh fixes cross_refresh_rms exactly at 1.0")
    return {
        "enabled": True,
        "schema": SCHEMA,
        "arm": arm,
        "ledger_path": ledger_path,
        "zero_all_return_biases": zero_all,
        "freeze_zeroed_biases": freeze_zeroed,
        "homogeneous_first_operator": int(raw.get("homogeneous_first_operator", 0)),
        "centered_value_gain_max": float(raw.get("centered_value_gain_max", 1.0e4)),
        "cross_refresh_rms": cross_refresh_rms,
    }


@dataclass
class CenteredValueState:
    handle: torch.utils.hooks.RemovableHandle | None
    latent_base: torch.Tensor
    gain_max: float
    gain: float | None = None
    calibration: dict[str, Any] | None = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


@dataclass
class RuntimeState:
    config: dict[str, Any]
    startup_ledger: dict[str, Any]
    centered_value: CenteredValueState | None = None


def _install_centered_value_hook(root: nn.Module, gain_max: float) -> CenteredValueState:
    bridge = getattr(root, "mandatory_latent_bridge", None)
    latent_base = getattr(root, "latent_base", None)
    if bridge is None or not isinstance(getattr(bridge, "v_proj", None), nn.Linear):
        raise TypeError("centered_value requires mandatory_latent_bridge.v_proj nn.Linear")
    if not isinstance(latent_base, torch.Tensor) or latent_base.ndim != 2:
        raise TypeError("centered_value requires model.latent_base [L,D]")
    if not math.isfinite(gain_max) or gain_max <= 0.0:
        raise ValueError("centered_value_gain_max must be finite and positive")

    state: CenteredValueState

    def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(args) != 1 or args[0].ndim != 3:
            raise RuntimeError("mandatory_latent_bridge.v_proj input contract changed")
        z = args[0]
        if tuple(z.shape[1:]) != tuple(state.latent_base.shape):
            raise RuntimeError(
                "centered_value latent shape mismatch: "
                f"{tuple(z.shape[1:])} != {tuple(state.latent_base.shape)}"
            )
        delta = z - state.latent_base.to(device=z.device, dtype=z.dtype).unsqueeze(0)
        if state.gain is None:
            linear = module
            assert isinstance(linear, nn.Linear)
            with torch.no_grad():
                original_v = F.linear(z.float(), linear.weight.float(), None)
                centered_v = F.linear(delta.float(), linear.weight.float(), None)
                original_rms = original_v.square().mean().sqrt()
                centered_rms = centered_v.square().mean().sqrt()
                if not torch.isfinite(original_rms) or not torch.isfinite(centered_rms):
                    raise RuntimeError("centered_value calibration RMS is non-finite")
                if float(centered_rms.item()) <= 0.0:
                    raise RuntimeError("centered_value calibration denominator is zero")
                gain = float((original_rms / centered_rms).item())
                if not math.isfinite(gain) or not 0.0 < gain <= state.gain_max:
                    raise RuntimeError(
                        f"centered_value calibrated gain {gain} is outside (0, {state.gain_max}]"
                    )
                state.gain = gain
                state.calibration = {
                    "gain": gain,
                    "original_v_rms": float(original_rms.item()),
                    "centered_v_rms_before_gain": float(centered_rms.item()),
                    "calibration_batch": int(z.shape[0]),
                    "latent_shape": list(z.shape[1:]),
                }
        assert state.gain is not None
        return (delta * state.gain,)

    placeholder = CenteredValueState(handle=None, latent_base=latent_base, gain_max=float(gain_max))
    state = placeholder
    state.handle = bridge.v_proj.register_forward_pre_hook(hook)
    return state


def prepare_model(model: nn.Module, raw_config: Mapping[str, Any]) -> RuntimeState:
    config = validate_config(raw_config)
    if not config["enabled"]:
        return RuntimeState(config=config, startup_ledger={"enabled": False})
    root = _unwrap_model(model)
    arm = config["arm"]
    architecture = str(getattr(root, "architecture_version", ""))
    expected_architecture = (
        "latent_mandatory_cross_refresh_posfilm_v7"
        if config["arm"] == "mandatory_cross_refresh"
        else (
            "clean_content_readout_posfilm_v8"
            if config["arm"] == "clean_content_readout"
            else "latent_mandatory_bridge_posfilm_v6"
        )
    )
    if architecture != expected_architecture:
        raise ValueError(
            f"causal arm {config['arm']!r} requires architecture {expected_architecture!r}, got {architecture!r}"
        )
    if arm == "mandatory_cross_refresh" and float(root.mandatory_cross_refresh_rms) != float(
        config["cross_refresh_rms"]
    ):
        raise RuntimeError("V7 model/config cross_refresh_rms mismatch")

    versions_before = {name: int(parameter._version) for name, parameter in root.named_parameters()}
    intended: list[str] = []
    before: dict[str, dict[str, Any]] = {}
    centered_state: CenteredValueState | None = None
    if arm == "zero_common_bias":
        intended.append("patch_tokenizer.y_proj.bias")
        if config["zero_all_return_biases"]:
            return_biases = _return_projection_bias_names(root)
            if not return_biases:
                raise RuntimeError("zero_all_return_biases resolved no residual return projections")
            intended.extend(return_biases)
        intended = sorted(set(intended))
        parameters = dict(root.named_parameters())
        missing = sorted(set(intended) - set(parameters))
        if missing:
            raise RuntimeError(f"V6 causal mutation target parameters are missing: {missing}")
        before = {name: _tensor_record(parameters[name]) for name in intended}
        with torch.no_grad():
            for name in intended:
                parameters[name].zero_()
        if config["freeze_zeroed_biases"]:
            for name in intended:
                parameters[name].requires_grad_(False)
    elif arm == "centered_value":
        centered_state = _install_centered_value_hook(root, config["centered_value_gain_max"])

    architectural_frozen: list[str] = []
    architectural_zero_biases: dict[str, dict[str, Any]] = {}
    if arm == "mandatory_cross_refresh":
        encoder_layers = getattr(root, "encoder_layers", None)
        if not isinstance(encoder_layers, nn.ModuleList) or not encoder_layers:
            raise RuntimeError("mandatory_cross_refresh requires non-empty encoder_layers")
        parameters = dict(root.named_parameters())
        for layer_index, layer in enumerate(encoder_layers):
            block = getattr(layer, "perceiver_block", None)
            local_names = getattr(block, "mandatory_cross_refresh_frozen_parameter_names", ())
            if not local_names:
                raise RuntimeError(f"encoder layer {layer_index} lacks mandatory cross-refresh freeze ledger")
            for local_name in local_names:
                full_name = f"encoder_layers.{layer_index}.perceiver_block.{local_name}"
                parameter = parameters.get(full_name)
                if parameter is None or parameter.requires_grad:
                    raise RuntimeError(f"mandatory cross-refresh parameter is not frozen: {full_name}")
                architectural_frozen.append(full_name)
                if local_name in {"norm_cross_kv.bias", "cross_v_proj.bias", "cross_out_proj.bias"}:
                    record = _tensor_record(parameter)
                    if record["nonzero_count"] != 0:
                        raise RuntimeError(f"mandatory cross-refresh value-path bias is nonzero: {full_name}")
                    architectural_zero_biases[full_name] = record
        feedback = getattr(root, "latent_to_weight_feedback", None)
        if not isinstance(feedback, nn.Module):
            raise RuntimeError("mandatory_cross_refresh requires preserved latent_to_weight_feedback module")
        for local_name, parameter in feedback.named_parameters():
            full_name = f"latent_to_weight_feedback.{local_name}"
            if parameter.requires_grad:
                raise RuntimeError(f"V7 bypassed latent feedback parameter is not frozen: {full_name}")
            architectural_frozen.append(full_name)

    v8_readout_contract: dict[str, Any] | None = None
    if arm == "clean_content_readout":
        readout = getattr(root, "clean_content_readout_v8", None)
        if not isinstance(readout, nn.Module):
            raise RuntimeError("clean_content_readout arm requires the V8 content readout")
        for name in ("q_proj", "k_proj", "out_proj"):
            projection = getattr(readout, name, None)
            if not isinstance(projection, nn.Linear) or projection.bias is not None:
                raise RuntimeError(f"V8 {name} must be a bias-free nn.Linear")
        frozen_names = tuple(getattr(root, "v8_bypassed_frozen_parameter_names", ()))
        if not frozen_names:
            raise RuntimeError("V8 bypass freeze ledger is empty")
        parameters = dict(root.named_parameters())
        missing = sorted(set(frozen_names) - set(parameters))
        still_trainable = sorted(name for name in frozen_names if parameters[name].requires_grad)
        if missing or still_trainable:
            raise RuntimeError(
                f"V8 bypass freeze mismatch: missing={missing} trainable={still_trainable}"
            )
        architectural_frozen.extend(frozen_names)
        with torch.no_grad():
            expected_routes = int(readout.num_latents * readout.n_heads)
            expected_rank = min(expected_routes, 1024)
            fixed_routing = torch.softmax(
                readout._fixed_anchor_scores(torch.ones(1, 1024, dtype=torch.bool))[0],
                dim=-1,
            )
            entropy = -(fixed_routing * fixed_routing.clamp_min(1.0e-30).log()).sum(dim=-1)
            unique_argmax = int(torch.unique(fixed_routing.argmax(dim=-1)).numel())
            effective_keys_median = float(entropy.exp().median().item())
            max_mass_median = float(fixed_routing.amax(dim=-1).median().item())
        if (
            tuple(fixed_routing.shape) != (expected_routes, 1024)
            or unique_argmax != expected_rank
            or not 1.3 <= effective_keys_median <= 4.0
            or not 0.35 <= max_mass_median <= 0.9
        ):
            raise RuntimeError("V8 fixed production-shape routing failed the precommitted rank/locality gate")
        v8_readout_contract = {
            "formula": "O(softmax(Q(X,pos)K(X,pos)^T/sqrt(d))*V(raw_signed_W_patches))",
            "w_route": "raw_signed_patch_values_no_projection",
            "x_route": "qk_only",
            "return_norm_gate_residual_dropout": False,
            "position_kind": "unique_flattened_gaussian_route_anchors_sigma_0.6",
            "heads": int(readout.n_heads),
            "head_dim": int(readout.head_dim),
            "theoretical_w_to_z_rank": int(readout.num_latents * readout.n_heads * readout.patch_size),
            "strict_readout_invariants": ["zero", "fixed_x_scaling", "fixed_x_superposition"],
            "bridge_invariants": ["zero_only"],
            "bridge_scaling_superposition": "measured_not_asserted_softmax_kv",
            "local_init_seed": 8008,
            "fixed_production_shape_routing": {
                "keys": 1024,
                "routes": expected_routes,
                "unique_argmax": unique_argmax,
                "full_rank_gate": {
                    "test": "tests/weightclip_benchmark/test_ae_architecture_v8.py::test_clean_readout_production_shape_anchor_routing_is_narrow_unique_and_full_rank",
                    "expected_empirical_rank_rtol_1e-6": 768,
                    "expected_stable_rank": 481.15185546875,
                },
                "effective_keys_median": effective_keys_median,
                "max_mass_median": max_mass_median,
            },
        }

    versions_after = {name: int(parameter._version) for name, parameter in root.named_parameters()}
    changed = sorted(name for name in versions_before if versions_before[name] != versions_after[name])
    if changed != intended:
        raise RuntimeError(f"V6 causal mutation set mismatch: intended={intended} actual={changed}")
    parameters = dict(root.named_parameters())
    after = {name: _tensor_record(parameters[name]) for name in intended}
    for name in intended:
        if after[name]["nonzero_count"] != 0:
            raise RuntimeError(f"V6 causal mutation failed to zero {name}")

    ledger = {
        "schema": SCHEMA,
        "arm": arm,
        "fresh_model_only": True,
        "model_architecture_version": str(root.architecture_version),
        "mandatory_cross_refresh_rms": (
            float(root.mandatory_cross_refresh_rms) if arm == "mandatory_cross_refresh" else None
        ),
        "zero_all_return_biases": bool(config["zero_all_return_biases"]),
        "freeze_zeroed_biases": bool(config["freeze_zeroed_biases"]),
        "frozen_zero_parameter_count": int(
            sum(parameters[name].numel() for name in intended)
            if config["freeze_zeroed_biases"]
            else 0
        ),
        "intended_mutated_tensors": intended,
        "actual_version_changed_tensors": changed,
        "architectural_frozen_parameter_names": sorted(architectural_frozen),
        "architectural_frozen_parameter_count": int(
            sum(dict(root.named_parameters())[name].numel() for name in architectural_frozen)
        ),
        "architectural_zero_value_path_biases": architectural_zero_biases,
        "v8_clean_content_readout": v8_readout_contract,
        "mutations": {
            name: {"before": before[name], "after": after[name]}
            for name in intended
        },
        "centered_value": {
            "enabled": centered_state is not None,
            "projection": "mandatory_latent_bridge.v_proj",
            "input": "fixed_gain*(z-latent_base)",
            "keys_unchanged": True,
            "gain_calibration": "first_full_B18_fp32_rms_ratio_frozen",
        },
        "training_schedule": (
            {
                "kind": "cyclic_singleton",
                "selection": "source_position=(step-1)%18",
                "source_rows_per_step": 1,
                "duplication_factor": 18,
                "steps": 504,
                "selections_per_source_row": 28,
                "presentations_per_source_row": 504,
                "total_presentations": 9072,
                "effective_cumulative_objective_weight_per_source_row": 28.0,
                "mixed_500_reference_objective_weight_per_source_row": 500.0 / 18.0,
                "physical_duplicates_are_independent_samples": False,
                "causal_scope": "singleton_stochastic_adam_path_and_intra_cycle_interference_jointly",
            }
            if arm == "cyclic_singleton"
            else {
                "kind": arm,
                "steps": 500,
                "physical_batch": 18,
                "causal_scope": (
                    "mandatory_refresh_plus_removed_common_self_state_plus_bias_free_value_path_jointly"
                    if arm == "mandatory_cross_refresh"
                    else (
                        "clean_w_value_x_position_routing_single_readout_jointly"
                        if arm == "clean_content_readout"
                        else arm
                    )
                ),
            }
        ),
    }
    return RuntimeState(config=config, startup_ledger=ledger, centered_value=centered_state)


def validate_v8_step0_preflight(
    model: nn.Module,
    metrics: Mapping[str, float],
    *,
    adam_eps: float,
) -> dict[str, Any]:
    """Fail before optimizer step 1 if the exact B18 V8 path is degenerate."""

    root = _unwrap_model(model)
    if not bool(getattr(root, "use_clean_content_readout_v8", False)):
        raise ValueError("V8 step-0 preflight requires clean_content_readout_posfilm_v8")
    if not math.isfinite(float(adam_eps)) or float(adam_eps) <= 0.0:
        raise ValueError("V8 step-0 preflight requires positive finite Adam eps")
    required_metrics = (
        "identity_v8_z_rms",
        "identity_raw_z_paired_relative_delta",
        "identity_v8_z_w_swap_relative_delta",
        "identity_bridge_paired_relative_delta",
        "identity_qnorm_paired_relative_delta",
        "identity_v8_output_paired_relative_delta",
        "identity_v8_zero_z_rms",
        "identity_v8_readout_native_scaling_error",
        "identity_v8_readout_native_superposition_error",
        "identity_v8_readout_native_dtype_epsilon",
        "identity_v8_readout_fp32_tf32_disabled_scaling_error",
        "identity_v8_readout_fp32_tf32_disabled_superposition_error",
        "identity_v8_readout_fp32_tf32_disabled_backend_allow_tf32",
        "identity_v8_readout_fp32_tf32_disabled_backend_precision_highest",
        "identity_v8_bridge_zero_rms",
        "identity_v8_routing_effective_keys_median",
        "identity_v8_routing_max_mass_median",
        "identity_v8_routing_unique_argmax_min",
        "identity_v8_routing_raw_score_rms",
        "identity_v8_routing_bounded_score_rms",
        "identity_v8_routing_bounded_score_max_abs",
        "identity_v8_routing_bounded_score_saturation_fraction",
        "identity_v8_zero_w_output_rms",
    )
    missing = sorted(set(required_metrics) - set(metrics))
    if missing:
        raise RuntimeError(f"V8 step-0 evaluator metrics are incomplete: {missing}")
    measured = {name: float(metrics[name]) for name in required_metrics}
    if any(not math.isfinite(value) for value in measured.values()):
        raise RuntimeError("V8 step-0 evaluator contains non-finite metrics")

    parameter_names = (
        "clean_content_readout_v8.q_proj.weight",
        "clean_content_readout_v8.k_proj.weight",
        "clean_content_readout_v8.out_proj.weight",
        "mandatory_latent_bridge.v_proj.weight",
        "mandatory_latent_bridge.out_proj.weight",
        "direction_head.2.weight",
    )
    parameters = dict(root.named_parameters())
    gradient_rms: dict[str, float] = {}
    for name in parameter_names:
        parameter = parameters.get(name)
        if parameter is None or parameter.grad is None:
            raise RuntimeError(f"V8 step-0 required gradient is missing: {name}")
        value = float(parameter.grad.detach().float().square().mean().sqrt().item())
        if not math.isfinite(value):
            raise RuntimeError(f"V8 step-0 gradient is non-finite: {name}")
        gradient_rms[name] = value
    below_eps = {name: value for name, value in gradient_rms.items() if value <= float(adam_eps)}
    if below_eps:
        raise RuntimeError(f"V8 step-0 gradients do not clear Adam eps={adam_eps}: {below_eps}")
    if measured["identity_v8_z_rms"] <= 10.0 * float(adam_eps):
        raise RuntimeError("V8 step-0 z RMS does not clear 10x Adam eps")
    for name in (
        "identity_raw_z_paired_relative_delta",
        "identity_v8_z_w_swap_relative_delta",
        "identity_bridge_paired_relative_delta",
        "identity_qnorm_paired_relative_delta",
        "identity_v8_output_paired_relative_delta",
    ):
        if measured[name] <= 1.0e-6:
            raise RuntimeError(f"V8 step-0 identity path is degenerate: {name}={measured[name]}")
    if measured["identity_v8_zero_z_rms"] != 0.0 or measured["identity_v8_bridge_zero_rms"] != 0.0:
        raise RuntimeError("V8 step-0 zero W must produce exact zero z and bridge state")
    for name in (
        "identity_v8_readout_fp32_tf32_disabled_scaling_error",
        "identity_v8_readout_fp32_tf32_disabled_superposition_error",
    ):
        if measured[name] > 2.0e-5:
            raise RuntimeError(f"V8 step-0 FP32 linear readout invariant failed: {name}={measured[name]}")
    if measured["identity_v8_readout_fp32_tf32_disabled_backend_allow_tf32"] != 0.0:
        raise RuntimeError("V8 step-0 strict FP32 replay did not disable TF32")
    if measured["identity_v8_readout_fp32_tf32_disabled_backend_precision_highest"] != 1.0:
        raise RuntimeError("V8 step-0 strict FP32 replay did not use highest matmul precision")
    native_epsilon = measured["identity_v8_readout_native_dtype_epsilon"]
    native_tolerance = max(2.0e-5, 2.0 * native_epsilon)
    for name in (
        "identity_v8_readout_native_scaling_error",
        "identity_v8_readout_native_superposition_error",
    ):
        if measured[name] > native_tolerance:
            raise RuntimeError(
                "V8 step-0 native linear readout arithmetic exceeded dtype tolerance: "
                f"{name}={measured[name]} tolerance={native_tolerance}"
            )
    if not 1.3 <= measured["identity_v8_routing_effective_keys_median"] <= 4.0:
        raise RuntimeError("V8 step-0 routing effective-key count is outside [1.3,4.0]")
    if not 0.35 <= measured["identity_v8_routing_max_mass_median"] <= 0.9:
        raise RuntimeError("V8 step-0 routing max mass is outside [0.35,0.9]")
    if measured["identity_v8_routing_unique_argmax_min"] < 730.0:
        raise RuntimeError("V8 step-0 routing has fewer than 730 unique top-1 keys")
    if not 0.02 <= measured["identity_v8_routing_bounded_score_rms"] <= 0.08:
        raise RuntimeError("V8 step-0 bounded X-routing score RMS is outside [0.02,0.08]")
    if measured["identity_v8_routing_bounded_score_max_abs"] > 0.10001:
        raise RuntimeError("V8 step-0 bounded X-routing score exceeds 0.10001")
    saturation_fraction = measured["identity_v8_routing_bounded_score_saturation_fraction"]
    if not 0.0 <= saturation_fraction <= 1.0:
        raise RuntimeError("V8 step-0 bounded X-routing saturation fraction is invalid")
    dtype_by_epsilon = {
        float(torch.finfo(dtype).eps): str(dtype).removeprefix("torch.")
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    }
    native_dtype = dtype_by_epsilon.get(native_epsilon)
    if native_dtype is None:
        raise RuntimeError(f"V8 step-0 native dtype epsilon is unrecognized: {native_epsilon}")
    return {
        "schema": "weightclip_ae_v8_exact_b18_step0_preflight_v2",
        "status": "pass",
        "optimizer_step_applied": False,
        "physical_batch": 18,
        "raw_value_projection": "identity_no_parameters",
        "adam_eps": float(adam_eps),
        "gradient_rms": gradient_rms,
        "gradient_to_adam_eps_ratio": {
            name: value / float(adam_eps) for name, value in gradient_rms.items()
        },
        "fp32_tf32_disabled_readout_arithmetic": {
            "dtype": "float32",
            "epsilon": float(torch.finfo(torch.float32).eps),
            "tolerance": 2.0e-5,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
        "native_readout_arithmetic": {
            "dtype": native_dtype,
            "epsilon": native_epsilon,
            "tolerance": native_tolerance,
        },
        "metrics": measured,
    }


def homogeneous_training_indices(
    logical_indices: Sequence[int],
    *,
    step: int,
    first_operator: int = 0,
) -> tuple[int, tuple[int, ...]]:
    if len(logical_indices) != 18 or len(set(int(value) for value in logical_indices)) != 18:
        raise ValueError("alternating_homogeneous requires one unique 18-tile logical cycle")
    operator = (int(first_operator) + int(step) - 1) % 2
    selected = tuple(index for index, logical in enumerate(logical_indices) if int(logical) % 2 == operator)
    if len(selected) != 9:
        raise RuntimeError(f"alternating_homogeneous expected nine tiles for operator {operator}, got {len(selected)}")
    duplicate = tuple(index for index in selected for _ in range(2))
    return operator, duplicate


def transform_training_tensors(
    tensors: Sequence[torch.Tensor],
    logical_indices: Sequence[int],
    *,
    step: int,
    arm: str = "alternating_homogeneous",
    first_operator: int = 0,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    if len(tensors) != 5 or any(int(tensor.shape[0]) != 18 for tensor in tensors):
        raise ValueError("alternating_homogeneous requires five tensors with batch dimension 18")
    if arm == "alternating_homogeneous":
        operator, indices = homogeneous_training_indices(
            logical_indices,
            step=step,
            first_operator=first_operator,
        )
        ledger = {
            "arm": arm,
            "step": int(step),
            "selected_operator": int(operator),
            "source_unique_tiles": 9,
            "effective_batch_rows": 18,
            "duplication_factor": 2,
            "source_positions": list(indices[::2]),
        }
    elif arm == "cyclic_singleton":
        if len(logical_indices) != 18 or len(set(int(value) for value in logical_indices)) != 18:
            raise ValueError("cyclic_singleton requires one unique 18-tile logical cycle")
        source_position = (int(step) - 1) % 18
        indices = (source_position,) * 18
        ledger = {
            "arm": arm,
            "step": int(step),
            "selected_source_position": source_position,
            "selected_logical_index": int(logical_indices[source_position]),
            "selected_operator": int(source_position % 2),
            "source_unique_tiles": 1,
            "effective_batch_rows": 18,
            "duplication_factor": 18,
        }
    else:
        raise ValueError(f"unsupported batch-transform arm {arm!r}")
    index = torch.tensor(indices, device=tensors[0].device, dtype=torch.long)
    transformed = tuple(tensor.index_select(0, index) for tensor in tensors)
    return transformed, ledger


def evaluation_due(*, step: int, every_steps: int, final_step: int) -> bool:
    if int(step) < 1 or int(every_steps) < 1 or int(final_step) < 1:
        raise ValueError("evaluation schedule values must be positive")
    return int(step) == 1 or int(step) % int(every_steps) == 0 or int(step) == int(final_step)


__all__ = [
    "ARMS",
    "SCHEMA",
    "RuntimeState",
    "evaluation_due",
    "homogeneous_training_indices",
    "prepare_model",
    "transform_training_tensors",
    "validate_config",
]
