"""Canonical Hydra composition for WeightCLIP AE architecture candidates.

PyYAML is intentionally not a scientific-config resolver.  In particular it
parses some scientific-notation scalars differently from OmegaConf.  Every AE
fingerprint, parameter ledger, approval, production launch, and bounded runtime
profile therefore goes through the same Hydra composition implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from big_vae.weightclip_benchmark.ae_scaling import canonical_fingerprint
from big_vae.weightclip_benchmark.manifests import sha256_file
from big_vae.weightclip_benchmark.metadata import redact_secrets


HYDRA_TRAIN_CONFIG_NAME = "big_vae/train/default"
ALLOWED_PRODUCTION_OPERATIONAL_OVERRIDE_PREFIXES = (
    "train.telemetry.comet.experiment_name=",
    "train.telemetry.wandb.run_name=",
)
BOUNDED_PROFILE_ONLY_CONFIG_PATHS = (
    "weightclip_runtime_profile_preflight",
    "train.bounded_runtime_profile",
    "train.checkpoint_dir",
    "train.resume_state.enabled",
    "train.resume_state.auto_resume",
    "train.resume_state.dir",
    "train.resume_state.explicit_checkpoint",
    "train.telemetry.comet.experiment_name",
    "train.telemetry.wandb.run_name",
    "train.telemetry.comet.enabled",
    "train.telemetry.wandb.enabled",
)
AE_IMPLEMENTATION_PATHS = (
    "big_vae/entrypoints/train.py",
    "big_vae/weightclip_benchmark/ae_candidate_config.py",
    "big_vae/weightclip_benchmark/ae_scaling.py",
    "big_vae/weightclip_benchmark/manifests.py",
    "big_vae/weightclip_benchmark/metadata.py",
    "big_vae/weightclip_benchmark/parameter_adapters.py",
    "experiments/background_prefetch.py",
    "training/forensics.py",
    "training/optim.py",
    "training/runtime.py",
    "training/weightclip_benchmark/build_operator_dataset.py",
    "training/weightclip_benchmark/profile_ae.py",
    "training/weightclip_benchmark/profile_ae_runtime.py",
    "training/weightclip_benchmark/compare_ae_runtime_profiles.py",
    "training/weightclip_benchmark/prepare_ae_approval_request.py",
    "training/weightclip_benchmark/approve_ae.py",
    "training/weightclip_benchmark/seal_ae.py",
    "training/weightclip_benchmark/train_ae.py",
)
AE_IMPLEMENTATION_GLOBS = (
    "big_vae/datasets/**/*.py",
    "big_vae/models/**/*.py",
    "dataset/**/*.py",
    "training/big_vae/**/*.py",
)


@dataclass(frozen=True, slots=True)
class ResolvedAECandidate:
    resolved_model_config: dict[str, Any]
    hydra_model_overrides: tuple[str, ...]
    model_fingerprint_sha256: str


def ae_source_implementation_seal(*, workspace_root: Path | None = None) -> dict[str, Any]:
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    paths = list(AE_IMPLEMENTATION_PATHS)
    for pattern in AE_IMPLEMENTATION_GLOBS:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise RuntimeError(f"AE source implementation glob resolved no files: {pattern}")
        paths.extend(str(path.relative_to(root)) for path in matches)
    paths = sorted(set(paths))
    files = []
    for relative in sorted(paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"AE source implementation file is missing: {path}")
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    contract = {"schema_version": 1, "kind": "weightclip_ae_source_implementation", "files": files}
    return {**contract, "source_implementation_seal_sha256": canonical_fingerprint(contract)}


def validate_ae_source_implementation_seal(
    seal: Mapping[str, Any], *, workspace_root: Path | None = None
) -> str:
    actual = ae_source_implementation_seal(workspace_root=workspace_root)
    if dict(seal) != actual:
        raise RuntimeError(
            "AE source implementation seal differs from the current model/worker/loss/data source inventory"
        )
    return str(actual["source_implementation_seal_sha256"])


def hydra_model_overrides(candidate_seed: Mapping[str, Any]) -> list[str]:
    """Translate the explicit scaling axes into production Hydra overrides."""

    model = candidate_seed.get("model", candidate_seed)
    big = model["big_vae"]
    dist = big["distribution_encoder"]
    tokenizer = big["patch_tokenizer"]
    values = {
        "model.big_vae.d_model": big["d_model"],
        "model.big_vae.d_lat": big["d_lat"],
        "model.big_vae.num_latents": big["num_latents"],
        "model.big_vae.num_encoder_layers": big["num_encoder_layers"],
        "model.big_vae.num_decoder_layers": big["num_decoder_layers"],
        "model.big_vae.n_heads": big["n_heads"],
        "model.big_vae.ffn_mult": big["ffn_mult"],
        "model.big_vae.use_latent_sampling": False,
        "model.big_vae.use_encoder_mu_head": False,
        "model.big_vae.latent_prior_kind": "gaussian",
        "model.big_vae.disable_z_shortcut": True,
        "model.big_vae.distribution_encoder.d_var": dist["d_var"],
        "model.big_vae.distribution_encoder.d_dist": dist["d_dist"],
        "model.big_vae.distribution_encoder.num_var_attn_layers": dist["num_var_attn_layers"],
        "model.big_vae.patch_tokenizer.d_patch": tokenizer["d_patch"],
    }
    if tokenizer.get("hidden_dim") is not None:
        values["model.big_vae.patch_tokenizer.hidden_dim"] = tokenizer["hidden_dim"]
    if big.get("decoder_query_conditioning_hidden_mult") is not None:
        values["model.big_vae.decoder_query_conditioning_hidden_mult"] = big[
            "decoder_query_conditioning_hidden_mult"
        ]
    return [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in values.items()]


def compose_resolved_train_config(
    overrides: Sequence[str], *, config_dir: Path | None = None
) -> dict[str, Any]:
    """Compose exactly the config consumed by ``big_vae.entrypoints.train``."""

    resolved_config_dir = (
        Path(config_dir).resolve()
        if config_dir is not None
        else Path(__file__).resolve().parents[2] / "conf"
    )
    with initialize_config_dir(version_base=None, config_dir=str(resolved_config_dir)):
        composed = compose(config_name=HYDRA_TRAIN_CONFIG_NAME, overrides=list(overrides))
    # Application logging references Hydra's runtime-only job object and a
    # timestamp.  It is not part of the model/scientific contract and cannot be
    # resolved through Compose outside an executing Hydra job.
    unresolved = OmegaConf.to_container(composed, resolve=False)
    if not isinstance(unresolved, dict):
        raise TypeError("Hydra production config did not compose to a mapping")
    unresolved.pop("logging", None)
    payload = OmegaConf.to_container(OmegaConf.create(unresolved), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra production config did not resolve to a mapping")
    return payload


def resolved_model_config(composed: Mapping[str, Any]) -> dict[str, Any]:
    model = composed.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Hydra-composed training config lacks a model mapping")
    return OmegaConf.to_container(OmegaConf.create(dict(model)), resolve=True)  # type: ignore[return-value]


def resolve_ae_candidate(candidate_seed: Mapping[str, Any]) -> ResolvedAECandidate:
    """Return the only model mapping which may be fingerprinted or profiled."""

    overrides = tuple(hydra_model_overrides(candidate_seed))
    model = resolved_model_config(compose_resolved_train_config(overrides))
    return ResolvedAECandidate(
        resolved_model_config=model,
        hydra_model_overrides=overrides,
        model_fingerprint_sha256=canonical_fingerprint(model),
    )


def assert_exact_candidate_model(
    expected: ResolvedAECandidate,
    composed: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Fail closed if a launch/profile resolves a different model mapping."""

    actual = resolved_model_config(composed)
    actual_fingerprint = canonical_fingerprint(actual)
    if actual != expected.resolved_model_config or actual_fingerprint != expected.model_fingerprint_sha256:
        raise RuntimeError(
            f"{context} Hydra model differs from the canonical candidate: "
            f"expected={expected.model_fingerprint_sha256} actual={actual_fingerprint}"
        )


def _remove_dotted(payload: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor.pop(parts[-1], None)


def normalize_launcher_runtime_config(composed: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields created or rewritten by the runtime launcher.

    Hydra composition cannot resolve ``logging`` outside a running Hydra job,
    and ``configure_per_run_artifacts`` creates ``training_artifacts`` after
    composition.  It also rewrites the exact output-path fields below through
    interpolations rooted at ``training_artifacts``.  The allowlist is derived
    from a real bounded launch; no model, loss, optimizer, scheduler, data
    selection, or numeric training field is normalized here.
    """

    payload = copy.deepcopy(dict(composed))
    payload.pop("logging", None)
    payload.pop("training_artifacts", None)
    for dotted in (
        "collector.diagnostics.crash_report_path",
        "collector.diagnostics.worker_status_dir",
        "train.fixed_training_batch.dump_path",
        "train.offline_dataset.analysis.output_dir",
        "train.offline_dataset.root_dir",
        "train.preslicing.root_dir",
        "train.resume_state.dir",
        "train.resume_state.explicit_checkpoint",
        "train.telemetry.grad_layer_monitor.csv_path",
        "train.telemetry.grad_layer_monitor.heatmap_path",
        "train.telemetry.grad_layer_monitor.plot_path",
        "train.telemetry.wandb.dir",
    ):
        _remove_dotted(payload, dotted)
    return payload


def production_scientific_config_payload(composed: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only explicitly operational output/run-label fields."""

    payload = normalize_launcher_runtime_config(composed)
    for dotted in (
        "weightclip_launch_preflight",
        "train.checkpoint_dir",
        "train.resume_state.explicit_checkpoint",
        "train.telemetry.comet.experiment_name",
        "train.telemetry.wandb.run_name",
    ):
        _remove_dotted(payload, dotted)
    redacted = redact_secrets(payload)
    if not isinstance(redacted, dict):
        raise TypeError("redacted production scientific config is not a mapping")
    return redacted


def production_scientific_config_sha256(composed: Mapping[str, Any]) -> str:
    return canonical_fingerprint(production_scientific_config_payload(composed))


def validate_production_operational_overrides(overrides: Sequence[str]) -> None:
    forbidden = [
        value
        for value in overrides
        if not any(value.startswith(prefix) for prefix in ALLOWED_PRODUCTION_OPERATIONAL_OVERRIDE_PREFIXES)
    ]
    if forbidden:
        raise RuntimeError(
            "Production AE launch accepts only tracker run-name overrides; "
            f"scientific/train overrides are forbidden: {forbidden}"
        )


def assert_bounded_profile_production_parity(
    production: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    left = copy.deepcopy(dict(production))
    right = copy.deepcopy(dict(profile))
    for dotted in BOUNDED_PROFILE_ONLY_CONFIG_PATHS:
        _remove_dotted(left, dotted)
        _remove_dotted(right, dotted)
    left = redact_secrets(left)
    right = redact_secrets(right)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise TypeError("bounded runtime parity payloads must remain mappings after redaction")
    if left != right:
        def changed_paths(expected: Any, actual: Any, path: str = "") -> list[str]:
            if type(expected) is not type(actual):
                return [path or "<root>"]
            if isinstance(expected, dict):
                rows: list[str] = []
                for key in sorted(set(expected) | set(actual)):
                    child = f"{path}.{key}" if path else str(key)
                    if key not in expected or key not in actual:
                        rows.append(child)
                    else:
                        rows.extend(changed_paths(expected[key], actual[key], child))
                return rows
            if isinstance(expected, list):
                if len(expected) != len(actual):
                    return [path or "<root>"]
                rows = []
                for index, (expected_item, actual_item) in enumerate(
                    zip(expected, actual, strict=True)
                ):
                    rows.extend(changed_paths(expected_item, actual_item, f"{path}[{index}]"))
                return rows
            return [] if expected == actual else [path or "<root>"]

        paths = changed_paths(left, right)
        raise RuntimeError(
            "bounded runtime profile differs from production outside the safety allowlist: "
            f"changed_paths={paths[:64]} total={len(paths)}"
        )


__all__ = [
    "HYDRA_TRAIN_CONFIG_NAME",
    "ALLOWED_PRODUCTION_OPERATIONAL_OVERRIDE_PREFIXES",
    "AE_IMPLEMENTATION_PATHS",
    "AE_IMPLEMENTATION_GLOBS",
    "BOUNDED_PROFILE_ONLY_CONFIG_PATHS",
    "ResolvedAECandidate",
    "ae_source_implementation_seal",
    "assert_exact_candidate_model",
    "assert_bounded_profile_production_parity",
    "compose_resolved_train_config",
    "hydra_model_overrides",
    "normalize_launcher_runtime_config",
    "production_scientific_config_payload",
    "production_scientific_config_sha256",
    "resolve_ae_candidate",
    "resolved_model_config",
    "validate_production_operational_overrides",
    "validate_ae_source_implementation_seal",
]
