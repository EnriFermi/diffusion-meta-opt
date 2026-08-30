from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from big_vae.models.big_weight_vae import BigWeightVAE
from training.big_vae.model_config import build_big_vae_model_config


PRODUCTION_APPROVAL_SCOPE = "weightclip_ae_production_training"
PRODUCTION_APPROVAL_RECORD_SCHEMA = "weightclip_ae_explicit_user_approval_v1"


@dataclass(frozen=True, slots=True)
class ScalingAxes:
    """Explicit architectural choices which can spend the AE parameter budget."""

    name: str
    d_model: int
    d_lat: int
    num_latents: int
    num_encoder_layers: int
    num_decoder_layers: int
    n_heads: int
    ffn_mult: float = 4.0
    d_var: int | None = None
    d_dist: int | None = None
    distribution_layers: int | None = None
    patch_token_d: int | None = None
    patch_token_hidden: int | None = None
    decoder_query_hidden_mult: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScalingAxes":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown AE scaling axes: {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ApprovalRequirements:
    candidate_name: str
    model_fingerprint_sha256: str
    tile_rows: int
    tile_cols: int
    training_steps: int
    production_scientific_config_sha256: str | None = None
    candidate_artifact_set_sha256: str | None = None
    candidate_report_sha256: str | None = None
    candidate_index_sha256: str | None = None
    trainable_parameters: int | None = None
    runtime_profile_summary_sha256: str | None = None
    producer_stress_profile_summary_sha256: str | None = None
    pair_manifest_sha256: str | None = None
    source_implementation_seal_sha256: str | None = None
    approval_request_sha256: str | None = None
    approval_request_fingerprint_sha256: str | None = None
    canonical_config_sha256: str | None = None
    approved_config_sha256: str | None = None


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON/YAML-safe deep copy without OmegaConf containers."""

    if hasattr(value, "items"):
        return {str(key): _plain_value(child) for key, child in value.items()}
    raise TypeError(f"Expected mapping, got {type(value)!r}")


def _plain_value(value: Any) -> Any:
    if hasattr(value, "items"):
        return {str(key): _plain_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(child) for child in value]
    return value


def apply_scaling_axes(base_config: Mapping[str, Any], axes: ScalingAxes) -> dict[str, Any]:
    """Resolve a deterministic Weight-AE candidate without selecting it.

    The workshop production run is an AE pretrain.  KL/sampling is a separate
    later fine-tune, so the candidate fingerprint deliberately fixes the
    deterministic latent path here.
    """

    resolved = copy.deepcopy(_plain_mapping(base_config))
    model = resolved["model"] if isinstance(resolved.get("model"), dict) else resolved
    big = model.setdefault("big_vae", {})
    dist = big.setdefault("distribution_encoder", model.get("distribution", {}))
    tokenizer = big.setdefault("patch_tokenizer", {})

    big.update(
        {
            "d_model": int(axes.d_model),
            "d_lat": int(axes.d_lat),
            "num_latents": int(axes.num_latents),
            "num_encoder_layers": int(axes.num_encoder_layers),
            "num_decoder_layers": int(axes.num_decoder_layers),
            "n_heads": int(axes.n_heads),
            "ffn_mult": float(axes.ffn_mult),
            "use_latent_sampling": False,
            "use_encoder_mu_head": False,
            "latent_prior_kind": "gaussian",
            "disable_z_shortcut": True,
        }
    )
    if axes.decoder_query_hidden_mult is not None:
        big["decoder_query_conditioning_hidden_mult"] = float(axes.decoder_query_hidden_mult)
    if axes.d_var is not None:
        dist["d_var"] = int(axes.d_var)
    if axes.d_dist is not None:
        dist["d_dist"] = int(axes.d_dist)
    if axes.distribution_layers is not None:
        dist["num_var_attn_layers"] = int(axes.distribution_layers)
    if axes.patch_token_d is not None:
        tokenizer["d_patch"] = int(axes.patch_token_d)
    if axes.patch_token_hidden is not None:
        tokenizer["hidden_dim"] = int(axes.patch_token_hidden)
    validate_scaling_axes(axes)
    return resolved


def validate_scaling_axes(axes: ScalingAxes) -> None:
    positive = {
        "d_model": axes.d_model,
        "d_lat": axes.d_lat,
        "num_latents": axes.num_latents,
        "num_encoder_layers": axes.num_encoder_layers,
        "num_decoder_layers": axes.num_decoder_layers,
        "n_heads": axes.n_heads,
    }
    invalid = {key: value for key, value in positive.items() if int(value) <= 0}
    if invalid:
        raise ValueError(f"AE scaling values must be positive: {invalid}")
    if axes.d_model % axes.n_heads:
        raise ValueError(f"d_model={axes.d_model} must be divisible by n_heads={axes.n_heads}")
    if axes.d_lat % axes.n_heads:
        raise ValueError(f"d_lat={axes.d_lat} must be divisible by n_heads={axes.n_heads}")


def canonical_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_plain_mapping(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component_for_parameter(name: str) -> str:
    if name.startswith("distribution_encoder"):
        return "activation_distribution_encoder"
    if name.startswith("patch_tokenizer") or name.startswith("patch_token_proj"):
        return "weight_patch_tokenizer"
    if name.startswith(("encoder_layers", "enc_dist_", "encoder_conditioning_adapters")):
        return "weight_encoder"
    if name.startswith(("latent_base", "latent_norm", "to_mu", "to_logvar", "vamp_prior_base")):
        return "latent_bottleneck"
    if name.startswith(("latent_to_decoder", "pos_proj", "query_proj", "query_pos_proj", "decoder_layers")):
        return "weight_decoder"
    if name.startswith(("direction_head", "scale_head", "q_tokens_norm")):
        return "reconstruction_heads"
    if name.startswith("z_shortcut"):
        return "disabled_z_shortcut"
    if name.startswith("debug_"):
        return "debug_heads"
    return "other"


def exact_parameter_ledger(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Instantiate on ``meta`` and count exact registered/trainable parameters."""

    with torch.device("meta"):
        model = BigWeightVAE(build_big_vae_model_config(resolved_config))
    seen: set[int] = set()
    components: dict[str, dict[str, int]] = {}
    total = trainable = 0
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count = int(parameter.numel())
        component = _component_for_parameter(name)
        bucket = components.setdefault(component, {"total": 0, "trainable": 0, "frozen": 0})
        bucket["total"] += count
        total += count
        if parameter.requires_grad:
            bucket["trainable"] += count
            trainable += count
        else:
            bucket["frozen"] += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "components": dict(sorted(components.items())),
    }


def representation_rate_ledger(
    resolved_config: Mapping[str, Any],
    *,
    tile_rows: int,
    tile_cols: int,
    mean_valid_weight_scalars: float | None = None,
) -> dict[str, float | int]:
    cfg = build_big_vae_model_config(resolved_config)
    latent_scalars = int(cfg.big_vae.num_latents * cfg.big_vae.d_lat)
    full_tile_scalars = int(tile_rows * tile_cols)
    valid_scalars = (
        float(mean_valid_weight_scalars)
        if mean_valid_weight_scalars is not None
        else float(full_tile_scalars)
    )
    padded_rows = ((int(tile_rows) + cfg.patch_size - 1) // cfg.patch_size) * cfg.patch_size
    padded_cols = ((int(tile_cols) + cfg.patch_size - 1) // cfg.patch_size) * cfg.patch_size
    padded_scalars = int(padded_rows * padded_cols)
    return {
        "latent_scalars": latent_scalars,
        "valid_weight_scalars": valid_scalars,
        "full_tile_weight_scalars": full_tile_scalars,
        "padded_weight_scalars": padded_scalars,
        "valid_scalars_per_latent_scalar": valid_scalars / max(latent_scalars, 1),
        "full_tile_scalars_per_latent_scalar": full_tile_scalars / max(latent_scalars, 1),
        "padded_scalars_per_latent_scalar": padded_scalars / max(latent_scalars, 1),
    }


def optimizer_memory_lower_bound(parameter_ledger: Mapping[str, Any]) -> dict[str, float]:
    """Account only persistent parameters/gradients/Adam states, not activations."""

    n = int(parameter_ledger["trainable_parameters"])
    frozen = int(parameter_ledger["frozen_parameters"])
    gib = float(1024**3)
    return {
        "bf16_trainable_parameters_gib": n * 2 / gib,
        "fp32_frozen_parameters_gib": frozen * 4 / gib,
        "bf16_gradients_gib": n * 2 / gib,
        "fp32_adam_moments_gib": n * 8 / gib,
        "fp32_master_weights_if_used_gib": n * 4 / gib,
        "explicitly_excludes_activations_and_allocator_overhead": True,
    }


def build_candidate_profile(
    resolved_model_config: Mapping[str, Any],
    axes: ScalingAxes,
    *,
    tile_rows: int,
    tile_cols: int,
    target_trainable_parameters: int,
    mean_valid_weight_scalars: float | None = None,
) -> dict[str, Any]:
    """Profile an already Hydra-composed candidate model.

    Callers must use ``resolve_ae_candidate``.  Accepting a raw PyYAML base
    here previously made scalar parsing (for example ``1e-4``) disagree with
    the actual Hydra worker configuration.
    """

    resolved = _plain_mapping(resolved_model_config)
    parameter_ledger = exact_parameter_ledger(resolved)
    trainable = int(parameter_ledger["trainable_parameters"])
    return {
        "candidate": asdict(axes),
        "model_fingerprint_sha256": canonical_fingerprint(resolved),
        "resolved_model_config_sha256": canonical_fingerprint(resolved),
        "resolved_model_config": resolved,
        "parameter_ledger": parameter_ledger,
        "target_trainable_parameters": int(target_trainable_parameters),
        "relative_trainable_parameter_error": (trainable - target_trainable_parameters)
        / max(target_trainable_parameters, 1),
        "representation_rate": representation_rate_ledger(
            resolved,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            mean_valid_weight_scalars=mean_valid_weight_scalars,
        ),
        "optimizer_memory_lower_bound": optimizer_memory_lower_bound(parameter_ledger),
    }


def approval_template(requirements: ApprovalRequirements) -> dict[str, Any]:
    return {
        "status": "pending_user_approval",
        "approval_scope": PRODUCTION_APPROVAL_SCOPE,
        "approved_by_user": False,
        "candidate_name": requirements.candidate_name,
        "model_fingerprint_sha256": requirements.model_fingerprint_sha256,
        "tile_rows": int(requirements.tile_rows),
        "tile_cols": int(requirements.tile_cols),
        "training_steps": int(requirements.training_steps),
        "production_scientific_config_sha256": requirements.production_scientific_config_sha256,
        "candidate_artifact_set_sha256": requirements.candidate_artifact_set_sha256,
        "candidate_report_sha256": requirements.candidate_report_sha256,
        "candidate_index_sha256": requirements.candidate_index_sha256,
        "trainable_parameters": requirements.trainable_parameters,
        "runtime_profile_summary_sha256": requirements.runtime_profile_summary_sha256,
        "producer_stress_profile_summary_sha256": requirements.producer_stress_profile_summary_sha256,
        "pair_manifest_sha256": requirements.pair_manifest_sha256,
        "source_implementation_seal_sha256": requirements.source_implementation_seal_sha256,
        "approval_request_sha256": requirements.approval_request_sha256,
        "approval_request_fingerprint_sha256": requirements.approval_request_fingerprint_sha256,
        "canonical_config_sha256": requirements.canonical_config_sha256,
        "approved_config_sha256": requirements.approved_config_sha256,
        "note": "Set only after the user explicitly approves this exact architecture fingerprint.",
    }


def require_production_approval(path: str | Path, requirements: ApprovalRequirements) -> dict[str, Any]:
    approval_path = Path(path)
    if not approval_path.is_file():
        raise RuntimeError(
            f"Production AE launch blocked: approval file is missing: {approval_path}. "
            "Run the profiler and obtain explicit user approval first."
        )
    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    binding_fields = {
        "production_scientific_config_sha256": requirements.production_scientific_config_sha256,
        "candidate_artifact_set_sha256": requirements.candidate_artifact_set_sha256,
        "candidate_report_sha256": requirements.candidate_report_sha256,
        "candidate_index_sha256": requirements.candidate_index_sha256,
        "trainable_parameters": requirements.trainable_parameters,
        "runtime_profile_summary_sha256": requirements.runtime_profile_summary_sha256,
        "producer_stress_profile_summary_sha256": requirements.producer_stress_profile_summary_sha256,
        "pair_manifest_sha256": requirements.pair_manifest_sha256,
        "source_implementation_seal_sha256": requirements.source_implementation_seal_sha256,
        "approval_request_sha256": requirements.approval_request_sha256,
        "approval_request_fingerprint_sha256": requirements.approval_request_fingerprint_sha256,
        "canonical_config_sha256": requirements.canonical_config_sha256,
        "approved_config_sha256": requirements.approved_config_sha256,
    }
    missing_bindings = sorted(key for key, value in binding_fields.items() if value in (None, ""))
    if missing_bindings:
        raise RuntimeError(
            f"Production AE launch blocked: approval requirements lack immutable bindings: {missing_bindings}"
        )
    expected = {
        "schema": PRODUCTION_APPROVAL_RECORD_SCHEMA,
        "status": "approved",
        "approval_scope": PRODUCTION_APPROVAL_SCOPE,
        "approved_by_user": True,
        "production_launch_authorized": True,
        "candidate_name": requirements.candidate_name,
        "model_fingerprint_sha256": requirements.model_fingerprint_sha256,
        "tile_rows": int(requirements.tile_rows),
        "tile_cols": int(requirements.tile_cols),
        "training_steps": int(requirements.training_steps),
        **binding_fields,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Production AE launch blocked: approval mismatch: {mismatches}")
    return payload


__all__ = [
    "ApprovalRequirements",
    "PRODUCTION_APPROVAL_SCOPE",
    "PRODUCTION_APPROVAL_RECORD_SCHEMA",
    "ScalingAxes",
    "apply_scaling_axes",
    "approval_template",
    "build_candidate_profile",
    "canonical_fingerprint",
    "exact_parameter_ledger",
    "optimizer_memory_lower_bound",
    "representation_rate_ledger",
    "require_production_approval",
]
