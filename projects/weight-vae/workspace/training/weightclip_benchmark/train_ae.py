from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import yaml
import numpy as np

from big_vae.datasets.operator_bank import OperatorBankBundleDataset, OperatorBankTrainingDataset
from big_vae.weightclip_benchmark.ae_candidate_config import (
    assert_bounded_profile_production_parity,
    assert_exact_candidate_model,
    compose_resolved_train_config as _compose_resolved,
    hydra_model_overrides,
    normalize_launcher_runtime_config,
    production_scientific_config_sha256,
    resolve_ae_candidate,
    validate_production_operational_overrides,
    validate_ae_source_implementation_seal,
)
from big_vae.weightclip_benchmark.ae_scaling import (
    ApprovalRequirements,
    ScalingAxes,
    apply_scaling_axes,
    canonical_fingerprint,
    exact_parameter_ledger,
    require_production_approval,
)
from big_vae.weightclip_benchmark.manifests import (
    assert_file_snapshot,
    file_stat_identity,
    sha256_file,
    sha256_file_stable,
    write_json_immutable,
)


PRODUCTION_DISK_HEADROOM_FRACTION = 0.20
PRODUCTION_DISK_HEADROOM_MIN_BYTES = 20 * 1024**3
TORCH_SAVE_CONSERVATIVE_OVERHEAD_FRACTION = 0.05
PRODUCTION_CHECKPOINT_EVERY = 10_000
PRODUCTION_TRAINING_STEPS = 500_000
PRODUCTION_GPU_HEADROOM_FRACTION = 0.05
PRODUCTION_GPU_HEADROOM_MIN_MIB = 4096.0


def _expected_ae_state_schemas(
    resolved_model_config: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[tuple[int, ...], str]],
    list[tuple[str, tuple[int, ...], str, bool]],
]:
    """Build the approved architecture on meta and return exact state/parameter schemas."""

    from big_vae.models.big_weight_vae import BigWeightVAE
    from training.big_vae.model_config import build_big_vae_model_config

    with torch.device("meta"):
        model = BigWeightVAE(build_big_vae_model_config(resolved_model_config))
    state_schema = {
        str(key): (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for key, tensor in model.state_dict().items()
    }
    parameter_schema = [
        (
            str(key),
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for key, parameter in model.named_parameters()
    ]
    if not state_schema or not parameter_schema:
        raise RuntimeError("approved AE meta model produced an empty state/parameter schema")
    return state_schema, parameter_schema


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous().reshape(-1)
    raw = contiguous.view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _validate_model_state(
    raw_state: Any,
    *,
    expected_schema: Mapping[str, tuple[tuple[int, ...], str]],
    context: str,
) -> str:
    if not isinstance(raw_state, Mapping):
        raise RuntimeError(f"{context} model_state is not a mapping")
    if any(not isinstance(key, str) for key in raw_state):
        raise RuntimeError(f"{context} model_state contains a non-string key")
    actual_keys = {str(key) for key in raw_state}
    expected_keys = set(expected_schema)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{context} model_state keys differ: "
            f"missing={sorted(expected_keys - actual_keys)[:8]} "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )
    digest = hashlib.sha256()
    for key in sorted(expected_schema):
        tensor = raw_state[key]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{context} model_state[{key!r}] is not a tensor")
        actual = (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        if actual != expected_schema[key]:
            raise RuntimeError(
                f"{context} model_state[{key!r}] shape/dtype differs: "
                f"expected={expected_schema[key]} actual={actual}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"{context} model_state[{key!r}] contains NaN/Inf")
        tensor_sha = _tensor_content_sha256(tensor)
        digest.update(key.encode("utf-8"))
        digest.update(repr(actual).encode("utf-8"))
        digest.update(tensor_sha.encode("ascii"))
    return digest.hexdigest()


def _expected_operator_stream_contract(resolved_launch: Mapping[str, Any]) -> dict[str, Any]:
    train = resolved_launch["train"]
    operator = train["operator_bank"]
    payload = {
        "schema": "operator_bank_locality_stream_v1",
        "planning_algorithm": "global_stratum_sequence_exact_within_stratum_grouped_v1",
        "pair_manifest_sha256": str(operator["pair_manifest_sha256"]),
        "effective_seed": int(resolved_launch["data"].get("seed", 42)),
        "rank": 0,
        "world_size": 1,
        "repeat": bool(operator.get("repeat", True)),
        "permutation_views": bool(operator.get("permutation_views", True)),
        "canonical_probability": float(operator.get("canonical_probability", 1.0 / 6.0)),
        "max_active_strata": int(operator.get("max_active_strata", 256)),
        "max_active_bundle_bytes": int(
            operator.get("max_active_bundle_bytes", 1536 * 1024 * 1024)
        ),
        "slice_batch_size": int(train.get("slice_batch_size", 1)),
        "grad_accum_steps": int(train.get("grad_accum_steps", 1)),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "operator_bank_stream_contract_v1",
        "sha256": hashlib.sha256(body).hexdigest(),
        "payload": payload,
    }


def _expected_lr_at_step(train: Mapping[str, Any], step: int) -> float:
    base_lr = float(train.get("lr", 3e-4))
    maximum = max(1, int(train.get("max_steps", 1000)))
    warmup = max(0, int(train.get("warmup_steps", 100)))
    minimum = float(train.get("min_lr_ratio", 0.1))
    if warmup > 0 and step < warmup:
        multiplier = float(step + 1) / float(warmup)
    else:
        progress = (step - warmup) / float(max(1, maximum - warmup))
        progress = min(max(progress, 0.0), 1.0)
        multiplier = minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * multiplier


def _validate_optimizer_scheduler_scaler(
    checkpoint: Mapping[str, Any],
    *,
    parameter_schema: list[tuple[str, tuple[int, ...], str, bool]],
    resolved_launch: Mapping[str, Any],
    step: int,
) -> None:
    train = resolved_launch["train"]
    if str(train.get("optimizer_name", "adamw")).lower() != "adamw":
        raise RuntimeError("production resume requires the frozen AdamW optimizer")
    if str(train.get("scheduler_name", "cosine")).lower() not in {"cosine", "cosine_decay"}:
        raise RuntimeError("production resume requires the frozen cosine LambdaLR scheduler")
    if not bool(train.get("amp", True)):
        raise RuntimeError("production resume requires the frozen H100 BF16 AMP contract")
    if float(train.get("patch_tokenizer_alpha_lr", 0.0) or 0.0) != 0.0:
        raise RuntimeError("production resume requires the frozen single AdamW parameter group")
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"}:
        raise RuntimeError("resume optimizer_state must contain exact state/param_groups mappings")
    groups = optimizer["param_groups"]
    states = optimizer["state"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, Mapping):
        raise RuntimeError("resume optimizer_state must contain exactly one complete AdamW group")
    group = groups[0]
    expected_ids = list(range(len(parameter_schema)))
    expected_state_ids = {
        index
        for index, (_name, _shape, _dtype, requires_grad) in enumerate(parameter_schema)
        if requires_grad
    }
    expected_names = [[name for name, _shape, _dtype, _requires_grad in parameter_schema]]
    if checkpoint.get("optimizer_param_names") != expected_names:
        raise RuntimeError(
            "resume optimizer parameter-name order differs from the approved meta model"
        )
    if not isinstance(group, Mapping) or group.get("params") != expected_ids:
        raise RuntimeError("resume optimizer param-group IDs/cardinality differ from the approved model")
    if set(states) != expected_state_ids:
        raise RuntimeError("resume optimizer Adam state is incomplete or has unexpected parameter IDs")
    expected_lr = _expected_lr_at_step(train, step)
    scalar_expectations = {
        "lr": expected_lr,
        "initial_lr": float(train.get("lr", 3e-4)),
        "eps": float(train.get("eps", 1e-8)),
        "weight_decay": float(train.get("weight_decay", 0.01)),
    }
    for key, expected in scalar_expectations.items():
        actual = group.get(key)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise RuntimeError(
                f"resume optimizer param-group {key} differs: expected={expected} actual={actual}"
            )
    if tuple(float(value) for value in group.get("betas", ())) != tuple(
        float(value) for value in train.get("betas", (0.9, 0.95))
    ):
        raise RuntimeError("resume optimizer betas differ from the approved config")
    expected_flags = {
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": False,
        "fused": True,
        "decoupled_weight_decay": True,
    }
    mismatched_flags = {
        key: {"expected": expected, "actual": group.get(key)}
        for key, expected in expected_flags.items()
        if group.get(key) is not expected
    }
    if mismatched_flags:
        raise RuntimeError(f"resume optimizer implementation flags differ: {mismatched_flags}")
    for index, (_name, shape, dtype, requires_grad) in enumerate(parameter_schema):
        if not requires_grad:
            continue
        state = states[index]
        if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"resume optimizer state[{index}] is not a complete Adam state")
        state_step = state["step"]
        if (
            not isinstance(state_step, torch.Tensor)
            or state_step.ndim != 0
            or str(state_step.dtype) != "torch.float32"
            or not bool(torch.isfinite(state_step).item())
            or float(state_step.item()) != float(step)
        ):
            raise RuntimeError(f"resume optimizer state[{index}] has inconsistent step")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            if (
                not isinstance(moment, torch.Tensor)
                or tuple(moment.shape) != shape
                or str(moment.dtype) != dtype
                or not bool(torch.isfinite(moment).all().item())
            ):
                raise RuntimeError(
                    f"resume optimizer state[{index}].{moment_name} has wrong shape/dtype or NaN/Inf"
                )

    scheduler = checkpoint.get("scheduler_state")
    expected_scheduler_keys = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_is_initial",
        "_get_lr_called_within_step",
        "_last_lr",
        "lr_lambdas",
    }
    if not isinstance(scheduler, Mapping) or set(scheduler) != expected_scheduler_keys:
        raise RuntimeError("resume scheduler_state schema differs from the frozen LambdaLR runtime")
    base_lr = float(train.get("lr", 3e-4))
    if (
        scheduler.get("base_lrs") != [base_lr]
        or scheduler.get("last_epoch") != step
        or scheduler.get("_step_count") != step + 1
        or scheduler.get("_is_initial") is not False
        or scheduler.get("_get_lr_called_within_step") is not False
        or scheduler.get("lr_lambdas") != [None]
        or not isinstance(scheduler.get("_last_lr"), list)
        or len(scheduler["_last_lr"]) != 1
        or not math.isclose(
            float(scheduler["_last_lr"][0]), expected_lr, rel_tol=1e-12, abs_tol=1e-15
        )
    ):
        raise RuntimeError("resume scheduler_state values differ from the exact approved step")
    scaler = checkpoint.get("scaler_state")
    if scaler != {}:
        raise RuntimeError("H100 BF16 production resume requires an exact disabled-GradScaler state")


def _validate_rng_state(
    rng_state: Any,
    *,
    resolved_launch: Mapping[str, Any],
    step: int,
    expected_physical_gpu_uuid: str,
) -> None:
    if not isinstance(rng_state, Mapping) or set(rng_state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_active",
        "data_stream",
    }:
        raise RuntimeError("resume RNG payload is incomplete or has unexpected fields")
    python_state = rng_state["python"]
    numpy_state = rng_state["numpy"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or not isinstance(numpy_state, tuple)
        or len(numpy_state) != 5
        or not isinstance(numpy_state[1], np.ndarray)
        or numpy_state[1].dtype != np.uint32
        or numpy_state[1].shape != (624,)
    ):
        raise RuntimeError("resume Python/NumPy RNG state has the wrong schema")
    try:
        random_probe = __import__("random").Random()
        random_probe.setstate(python_state)
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(numpy_state)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("resume Python/NumPy RNG state cannot be restored") from exc
    torch_cpu = rng_state["torch_cpu"]
    torch_cuda_active = rng_state["torch_cuda_active"]
    expected_device = torch.device(str(resolved_launch["train"].get("device", "")))
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
        or expected_device.type != "cuda"
        or not isinstance(torch_cuda_active, Mapping)
        or set(torch_cuda_active)
        != {
            "schema",
            "logical_device",
            "physical_uuid",
            "device_name",
            "cuda_visible_devices",
            "state",
        }
        or torch_cuda_active.get("schema") != "active_cuda_rng_v1"
        or torch_cuda_active.get("logical_device") != str(expected_device)
        or torch_cuda_active.get("physical_uuid") != expected_physical_gpu_uuid
        or torch_cuda_active.get("cuda_visible_devices") != expected_physical_gpu_uuid
        or not isinstance(torch_cuda_active.get("state"), torch.Tensor)
        or torch_cuda_active["state"].dtype != torch.uint8
        or torch_cuda_active["state"].ndim != 1
        or torch_cuda_active["state"].numel() == 0
    ):
        raise RuntimeError("resume Torch CPU/CUDA RNG state has the wrong schema")
    _validate_torch_rng_states_restorable(
        torch_cpu,
        torch_cuda_active,
        expected_device=expected_device,
    )
    data_stream = rng_state["data_stream"]
    expected_contract = _expected_operator_stream_contract(resolved_launch)
    train = resolved_launch["train"]
    expected_logical_index = (
        step * int(train.get("grad_accum_steps", 1)) * int(train.get("slice_batch_size", 1))
    )
    if (
        not isinstance(data_stream, Mapping)
        or data_stream.get("exactly_restorable") is not True
        or data_stream.get("kind") != "operator_bank_committed_step_cursor"
        or data_stream.get("committed_training_step") != step
        or data_stream.get("logical_sample_index") != expected_logical_index
        or data_stream.get("operator_stream_contract") != expected_contract
    ):
        raise RuntimeError("resume RNG data-stream cursor/contract differs from the approved launch")


def _validate_torch_rng_states_restorable(
    torch_cpu: torch.Tensor,
    torch_cuda_active: Mapping[str, Any],
    *,
    expected_device: torch.device,
) -> None:
    """Validate RNG bytes on temporary generators without mutating process globals."""

    try:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.set_state(torch_cpu)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("resume Torch CPU RNG state cannot be restored") from exc
    try:
        _validate_cuda_rng_state_isolated(
            torch_cuda_active["state"],
            physical_gpu_uuid=str(torch_cuda_active["physical_uuid"]),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"resume Torch CUDA RNG state cannot be restored for {expected_device}"
        ) from exc


def _validate_cuda_rng_state_isolated(
    state: torch.Tensor,
    *,
    physical_gpu_uuid: str,
    runner: Any = subprocess.run,
) -> None:
    """Validate CUDA RNG bytes in a short UUID-masked child, never the launcher."""

    if torch.cuda.is_initialized():
        raise RuntimeError("launcher CUDA was initialized before isolated RNG validation")
    if not physical_gpu_uuid.startswith("GPU-"):
        raise RuntimeError("isolated RNG validation requires a canonical GPU-* UUID")
    raw_hex = state.detach().cpu().contiguous().numpy().tobytes().hex()
    script = (
        "import sys, torch\n"
        "from training.runtime import resolve_cuda_physical_identity\n"
        "expected=sys.argv[1]\n"
        "if torch.cuda.device_count() != 1:\n"
        "    raise RuntimeError('UUID mask did not expose exactly one CUDA device')\n"
        "identity=resolve_cuda_physical_identity(torch.device('cuda:0'))\n"
        "if identity['physical_uuid'] != expected:\n"
        "    raise RuntimeError('masked CUDA device UUID differs from approval')\n"
        "raw=bytes.fromhex(sys.stdin.read())\n"
        "state=torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()\n"
        "generator=torch.Generator(device='cuda:0')\n"
        "generator.set_state(state)\n"
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_uuid
    result = runner(
        [sys.executable, "-c", script, physical_gpu_uuid],
        input=raw_hex,
        text=True,
        capture_output=True,
        timeout=60.0,
        check=False,
        env=environment,
    )
    if int(result.returncode) != 0:
        stderr_tail = str(result.stderr or "")[-2000:]
        raise RuntimeError(f"isolated CUDA RNG validation failed: {stderr_tail}")
    if torch.cuda.is_initialized():
        raise RuntimeError("isolated CUDA RNG validation initialized the launcher process")


def _bind_parent_gpu_mask(physical_gpu_uuid: str) -> None:
    """Bind future CUDA children while keeping the launcher CUDA-uninitialized."""

    if torch.cuda.is_initialized():
        raise RuntimeError("production launcher CUDA was initialized before approved UUID masking")
    if not physical_gpu_uuid.startswith("GPU-"):
        raise RuntimeError("production launcher requires a canonical approved GPU-* UUID")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu_uuid


def _normalized_resume_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_launcher_runtime_config(config)
    normalized.pop("weightclip_launch_preflight", None)
    return normalized


def _validate_checkpoint_config(
    config: Any,
    *,
    resolved_launch: Mapping[str, Any],
    expected_preflight: Mapping[str, Any],
    checkpoint_root: str,
    resume_root: str,
    context: str,
) -> None:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context} embedded config is not a mapping")
    if canonical_fingerprint(_normalized_resume_config(config)) != canonical_fingerprint(
        _normalized_resume_config(resolved_launch)
    ):
        raise RuntimeError(f"{context} embedded resolved config differs from the approved launch")
    prior_preflight = config.get("weightclip_launch_preflight", {})
    stable_keys = {
        "candidate",
        "model_fingerprint_sha256",
        "candidate_config_sha256",
        "base_model_config_sha256",
        "resolved_model_config_sha256",
        "canonical_config_sha256",
        "approved_config_sha256",
        "approval_request_sha256",
        "approval_request_fingerprint_sha256",
        "approval_file_sha256",
        "production_scientific_config_sha256",
        "pair_manifest_sha256",
        "artifact_set_fingerprint_sha256",
        "candidate_report_sha256",
        "candidate_index_sha256",
        "runtime_profile_summary_sha256",
        "producer_stress_profile_summary_sha256",
        "source_implementation_seal_sha256",
        "trainable_parameters",
    }
    if not isinstance(prior_preflight, Mapping):
        raise RuntimeError(f"{context} lacks the prior launch preflight mapping")
    mismatches = {
        key: {"expected": expected_preflight.get(key), "actual": prior_preflight.get(key)}
        for key in sorted(stable_keys)
        if prior_preflight.get(key) != expected_preflight.get(key)
    }
    if mismatches:
        raise RuntimeError(f"{context} prior launch binding mismatch: {mismatches}")
    if (
        prior_preflight.get("checkpoint_root") != checkpoint_root
        or prior_preflight.get("resume_root") != resume_root
        or prior_preflight.get("launch_mode") not in {"fresh", "resume"}
    ):
        raise RuntimeError(f"{context} was produced under different storage roots")
    prior_train = config.get("train", {})
    prior_resume = prior_train.get("resume_state", {}) if isinstance(prior_train, Mapping) else {}
    if (
        Path(str(prior_train.get("checkpoint_dir", ""))).resolve() != Path(checkpoint_root)
        or Path(str(prior_resume.get("dir", ""))).resolve() != Path(resume_root)
    ):
        raise RuntimeError(f"{context} embedded config has different storage roots")


def _load_checkpoint_stable(path: Path, *, context: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or is a symlink: {path}")
    before = file_stat_identity(path)
    digest = sha256_file(path)
    middle = file_stat_identity(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    after = file_stat_identity(path)
    if before != middle or middle != after:
        raise RuntimeError(f"{context} changed during hash/load: {path}")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} is not a mapping: {path}")
    return payload, {"path": str(path), "sha256": digest, "stat": before}


def _validate_prior_model_archive(
    *,
    storage: Mapping[str, Any],
    step: int,
    expected_model_schema: Mapping[str, tuple[tuple[int, ...], str]],
    expected_resume_model_content_sha256: str,
    expected_preflight: Mapping[str, Any],
    expected_scientific_config_sha256: str,
    resolved_launch: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(str(storage["checkpoint_root"])) / "stage_1"
    expected_steps = list(range(PRODUCTION_CHECKPOINT_EVERY, step + 1, PRODUCTION_CHECKPOINT_EVERY))
    expected_paths = {root / f"step_{value:07d}.pt" for value in expected_steps}
    actual_paths = set(root.glob("step_*.pt"))
    if actual_paths != expected_paths:
        raise RuntimeError(
            "prior numbered model-checkpoint archive inventory differs: "
            f"missing={sorted(str(path) for path in expected_paths - actual_paths)[:8]} "
            f"unexpected={sorted(str(path) for path in actual_paths - expected_paths)[:8]}"
        )
    inventory: list[dict[str, Any]] = []
    final_numbered_content_sha256 = None
    for expected_step in expected_steps:
        path = root / f"step_{expected_step:07d}.pt"
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"numbered model checkpoint is not immutable/read-only: {path}")
        payload, snapshot = _load_checkpoint_stable(
            path, context=f"numbered model checkpoint step {expected_step}"
        )
        if set(payload) != {"step", "stage", "model_state", "config"}:
            raise RuntimeError(f"numbered model checkpoint payload schema differs: {path}")
        if payload.get("step") != expected_step or payload.get("stage") != 1:
            raise RuntimeError(f"numbered model checkpoint step/stage differs: {path}")
        _validate_checkpoint_config(
            payload["config"],
            resolved_launch=resolved_launch,
            expected_preflight=expected_preflight,
            checkpoint_root=str(storage["checkpoint_root"]),
            resume_root=str(storage["resume_root"]),
            context=f"numbered model checkpoint step {expected_step}",
        )
        if (
            production_scientific_config_sha256(payload["config"])
            != expected_scientific_config_sha256
        ):
            raise RuntimeError(f"numbered model checkpoint scientific config differs: {path}")
        model_content_sha256 = _validate_model_state(
            payload["model_state"],
            expected_schema=expected_model_schema,
            context=f"numbered model checkpoint step {expected_step}",
        )
        inventory.append(
            {
                **snapshot,
                "step": expected_step,
                "model_state_content_sha256": model_content_sha256,
            }
        )
        if expected_step == step:
            final_numbered_content_sha256 = model_content_sha256
        del payload
    latest_path = root / "latest.pt"
    latest, latest_snapshot = _load_checkpoint_stable(
        latest_path, context="latest model checkpoint"
    )
    if set(latest) != {"step", "stage", "model_state", "config"}:
        raise RuntimeError("latest model checkpoint payload schema differs")
    if latest.get("step") != step or latest.get("stage") != 1:
        raise RuntimeError("latest model checkpoint does not point at the resume step")
    _validate_checkpoint_config(
        latest["config"],
        resolved_launch=resolved_launch,
        expected_preflight=expected_preflight,
        checkpoint_root=str(storage["checkpoint_root"]),
        resume_root=str(storage["resume_root"]),
        context="latest model checkpoint",
    )
    if production_scientific_config_sha256(latest["config"]) != expected_scientific_config_sha256:
        raise RuntimeError("latest model checkpoint scientific config differs")
    latest_content_sha256 = _validate_model_state(
        latest["model_state"],
        expected_schema=expected_model_schema,
        context="latest model checkpoint",
    )
    if not (
        final_numbered_content_sha256
        == latest_content_sha256
        == expected_resume_model_content_sha256
    ):
        raise RuntimeError(
            "resume, final numbered, and latest model checkpoint states are not bitwise identical"
        )
    inventory.append(
        {
            **latest_snapshot,
            "step": step,
            "kind": "latest_alias",
            "model_state_content_sha256": latest_content_sha256,
        }
    )
    contract = {
        "schema": "weightclip_ae_prior_model_archive_v1",
        "checkpoint_root": str(storage["checkpoint_root"]),
        "resume_step": step,
        "expected_numbered_steps": expected_steps,
        "artifacts": inventory,
    }
    return {
        **contract,
        "archive_fingerprint_sha256": canonical_fingerprint(contract),
    }


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink_components(path: Path, *, context: str) -> Path:
    absolute = _absolute_path_without_symlink_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise RuntimeError(f"{context} contains a symlink component: {current}")
    return absolute


def _launch_storage_contract(
    *,
    launch_mode: str,
    checkpoint_root: Path,
    resume_state_checkpoint: Path | None,
) -> dict[str, Any]:
    if launch_mode not in {"fresh", "resume"}:
        raise RuntimeError(f"unsupported production launch mode: {launch_mode!r}")
    root = _reject_symlink_components(checkpoint_root, context="checkpoint root")
    if root == Path(root.anchor) or len(root.parts) < 3:
        raise RuntimeError(f"checkpoint root is too broad: {root}")
    resume_root = root / "stage_1" / "resume_state"
    if launch_mode == "fresh":
        if resume_state_checkpoint is not None:
            raise RuntimeError("fresh launch forbids --resume-state-checkpoint")
        if root.exists():
            raise RuntimeError(
                f"fresh launch requires a unique nonexistent checkpoint root: {root}"
            )
        if not root.parent.is_dir():
            raise RuntimeError(
                "fresh launch checkpoint-root parent must already exist for one atomic mkdir: "
                f"{root.parent}"
            )
        return {
            "launch_mode": launch_mode,
            "checkpoint_root": str(root),
            "resume_root": str(resume_root),
            "resume_state_checkpoint": None,
            "resume_state_checkpoint_sha256": None,
            "resume_state_checkpoint_snapshot": None,
            "resume_step": 0,
        }
    if resume_state_checkpoint is None:
        raise RuntimeError("resume launch requires --resume-state-checkpoint")
    checkpoint = _reject_symlink_components(
        resume_state_checkpoint, context="resume-state checkpoint"
    )
    if not root.is_dir() or not resume_root.is_dir():
        raise RuntimeError("resume launch requires existing checkpoint and resume roots")
    if checkpoint.parent != resume_root or not checkpoint.is_file():
        raise RuntimeError(
            "resume-state checkpoint must be an existing direct child of the exact resume root"
        )
    if checkpoint.stat().st_mode & 0o222:
        raise RuntimeError("resume-state checkpoint must be immutable/read-only")
    snapshot = sha256_file_stable(checkpoint)
    return {
        "launch_mode": launch_mode,
        "checkpoint_root": str(root),
        "resume_root": str(resume_root),
        "resume_state_checkpoint": str(checkpoint),
        "resume_state_checkpoint_sha256": snapshot["sha256"],
        "resume_state_checkpoint_snapshot": snapshot,
        "resume_step": None,
    }


def _storage_overrides(storage: Mapping[str, Any]) -> list[str]:
    explicit = storage.get("resume_state_checkpoint") or ""
    return [
        f"train.checkpoint_dir={storage['checkpoint_root']}",
        f"train.resume_state.dir={storage['resume_root']}",
        "train.resume_state.auto_resume=false",
        f"+train.resume_state.explicit_checkpoint={explicit}",
    ]


def _validate_resume_state_checkpoint(
    storage: dict[str, Any],
    *,
    expected_preflight: Mapping[str, Any],
    expected_scientific_config_sha256: str,
    resolved_launch: Mapping[str, Any],
    resolved_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    if storage["launch_mode"] != "resume":
        return storage
    snapshot = storage["resume_state_checkpoint_snapshot"]
    assert_file_snapshot(snapshot, context="production resume-state checkpoint")
    checkpoint_path = Path(str(storage["resume_state_checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    assert_file_snapshot(snapshot, context="production resume-state checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("resume-state checkpoint is not a mapping")
    required_payloads = {
        "step",
        "stage",
        "model_state",
        "optimizer_state",
        "optimizer_param_names",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "config",
    }
    if set(checkpoint) != required_payloads:
        raise RuntimeError(
            "resume-state checkpoint payload schema differs: "
            f"missing={sorted(required_payloads - set(checkpoint))} "
            f"unexpected={sorted(set(checkpoint) - required_payloads)}"
        )
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or step >= PRODUCTION_TRAINING_STEPS
        or step % PRODUCTION_CHECKPOINT_EVERY
    ):
        raise RuntimeError(
            "resume-state checkpoint step must be positive, below 500k, and aligned to 10k"
        )
    if checkpoint_path.name != f"step_{step:07d}.pt":
        raise RuntimeError("resume-state checkpoint filename does not match its embedded step")
    if checkpoint.get("stage") != 1:
        raise RuntimeError("resume-state checkpoint stage must be exactly 1")
    config = checkpoint["config"]
    _validate_checkpoint_config(
        config,
        resolved_launch=resolved_launch,
        expected_preflight=expected_preflight,
        checkpoint_root=str(storage["checkpoint_root"]),
        resume_root=str(storage["resume_root"]),
        context="resume-state checkpoint",
    )
    if production_scientific_config_sha256(config) != expected_scientific_config_sha256:
        raise RuntimeError("resume-state checkpoint scientific config differs from approval")
    expected_model_schema, parameter_schema = _expected_ae_state_schemas(resolved_model_config)
    resume_model_content_sha256 = _validate_model_state(
        checkpoint["model_state"],
        expected_schema=expected_model_schema,
        context="resume-state checkpoint",
    )
    _validate_optimizer_scheduler_scaler(
        checkpoint,
        parameter_schema=parameter_schema,
        resolved_launch=resolved_launch,
        step=step,
    )
    _validate_rng_state(
        checkpoint["rng_state"],
        resolved_launch=resolved_launch,
        step=step,
        expected_physical_gpu_uuid=str(expected_preflight["physical_gpu_uuid"]),
    )
    validated_storage = {**storage, "resume_step": step}
    archive = _validate_prior_model_archive(
        storage=validated_storage,
        step=step,
        expected_model_schema=expected_model_schema,
        expected_resume_model_content_sha256=resume_model_content_sha256,
        expected_preflight=expected_preflight,
        expected_scientific_config_sha256=expected_scientific_config_sha256,
        resolved_launch=resolved_launch,
    )
    return {
        **validated_storage,
        "prior_model_archive": archive,
        "prior_model_archive_fingerprint_sha256": archive["archive_fingerprint_sha256"],
        "prior_model_archive_artifact_count": len(archive["artifacts"]),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def _candidate(cfg: dict[str, Any], name: str) -> ScalingAxes:
    matches = [ScalingAxes.from_mapping(value) for value in cfg["candidates"] if value.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one candidate named {name!r}, found {len(matches)}")
    return matches[0]


def _hydra_model_overrides(resolved: dict[str, Any]) -> list[str]:
    """Compatibility wrapper around the canonical shared composer."""

    return hydra_model_overrides(resolved)


def _production_disk_preflight(
    *,
    resolved: dict[str, Any],
    parameter_ledger: dict[str, Any],
    report_root: Path,
    candidate_name: str,
    approved_config_sha256: str,
    storage: Mapping[str, Any],
    execute: bool,
) -> dict[str, Any]:
    """Conservatively bound checkpoint storage without allocating model tensors."""

    train = resolved["train"]
    training_steps = int(train["max_steps"])
    checkpoint_every = int(train["checkpoint_every"])
    resume_every = int(train["resume_state"]["save_every"])
    if (
        training_steps != PRODUCTION_TRAINING_STEPS
        or checkpoint_every != PRODUCTION_CHECKPOINT_EVERY
        or resume_every != PRODUCTION_CHECKPOINT_EVERY
    ):
        raise RuntimeError(
            "production disk preflight requires the frozen 500k/10k checkpoint cadence"
        )
    checkpoint_dir = Path(str(train["checkpoint_dir"])).expanduser().resolve()
    resume_dir = Path(str(train["resume_state"]["dir"])).expanduser().resolve()
    if (
        checkpoint_dir != Path(str(storage["checkpoint_root"]))
        or resume_dir != Path(str(storage["resume_root"]))
    ):
        raise RuntimeError("final resolved launch checkpoint/resume roots differ from storage contract")
    existing_parent = checkpoint_dir
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    usage = shutil.disk_usage(existing_parent)
    total_parameters = int(parameter_ledger["total_parameters"])
    trainable_parameters = int(parameter_ledger["trainable_parameters"])
    model_state_bytes = total_parameters * 4
    adam_moment_bytes = trainable_parameters * 8
    scheduled_steps = list(range(checkpoint_every, training_steps + 1, checkpoint_every))
    if not scheduled_steps or scheduled_steps[-1] != training_steps:
        scheduled_steps.append(training_steps)
    resume_step = int(storage["resume_step"])
    remaining_steps = [step for step in scheduled_steps if step > resume_step]
    scheduled_model_checkpoints = len(remaining_steps)
    # Worker persists every numbered model checkpoint plus latest.pt. Resume state is rolling,
    # while its atomic write temporarily coexists with the previous resume state.
    model_checkpoint_bytes = model_state_bytes
    resume_checkpoint_bytes = model_state_bytes + adam_moment_bytes
    if storage["launch_mode"] == "fresh":
        additional_latest_bytes = model_checkpoint_bytes
        additional_resume_bytes = resume_checkpoint_bytes
    else:
        additional_latest_bytes = 0
        additional_resume_bytes = 0
    persisted_core_bytes = (
        scheduled_model_checkpoints * model_checkpoint_bytes
        + additional_latest_bytes
        + additional_resume_bytes
    )
    atomic_transient_core_bytes = max(model_checkpoint_bytes, resume_checkpoint_bytes)
    serialization_overhead_bytes = int(
        (persisted_core_bytes + atomic_transient_core_bytes)
        * TORCH_SAVE_CONSERVATIVE_OVERHEAD_FRACTION
    )
    conservative_peak_bytes = (
        persisted_core_bytes + atomic_transient_core_bytes + serialization_overhead_bytes
    )
    headroom_bytes = max(
        PRODUCTION_DISK_HEADROOM_MIN_BYTES,
        int(conservative_peak_bytes * PRODUCTION_DISK_HEADROOM_FRACTION),
    )
    required_free_bytes = conservative_peak_bytes + headroom_bytes
    contract = {
        "schema": "weightclip_ae_production_disk_preflight_v1",
        "candidate_name": candidate_name,
        "approved_config_sha256": approved_config_sha256,
        "launch_mode": storage["launch_mode"],
        "checkpoint_dir": str(checkpoint_dir),
        "resume_dir": str(resume_dir),
        "resume_state_checkpoint": storage["resume_state_checkpoint"],
        "resume_state_checkpoint_sha256": storage["resume_state_checkpoint_sha256"],
        "resume_step": resume_step,
        "prior_model_archive": storage.get("prior_model_archive"),
        "prior_model_archive_fingerprint_sha256": storage.get(
            "prior_model_archive_fingerprint_sha256"
        ),
        "filesystem_probe_path": str(existing_parent),
        "filesystem_probe_stat": file_stat_identity(existing_parent),
        "training_steps": training_steps,
        "checkpoint_every": checkpoint_every,
        "resume_save_every": resume_every,
        "scheduled_numbered_model_checkpoints": scheduled_model_checkpoints,
        "remaining_numbered_checkpoint_steps": remaining_steps,
        "additional_latest_model_checkpoint_copies": int(additional_latest_bytes > 0),
        "additional_rolling_resume_checkpoint_copies": int(additional_resume_bytes > 0),
        "atomic_transient_resume_or_model_copies": 1,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "model_state_bytes_fp32": model_state_bytes,
        "adam_moment_bytes_fp32": adam_moment_bytes,
        "resume_checkpoint_core_bytes": resume_checkpoint_bytes,
        "persisted_core_bytes": persisted_core_bytes,
        "atomic_transient_core_bytes": atomic_transient_core_bytes,
        "serialization_overhead_fraction": TORCH_SAVE_CONSERVATIVE_OVERHEAD_FRACTION,
        "serialization_overhead_bytes": serialization_overhead_bytes,
        "conservative_peak_bytes": conservative_peak_bytes,
        "headroom_fraction": PRODUCTION_DISK_HEADROOM_FRACTION,
        "headroom_min_bytes": PRODUCTION_DISK_HEADROOM_MIN_BYTES,
        "headroom_bytes": headroom_bytes,
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": int(usage.free),
        "sufficient": int(usage.free) >= required_free_bytes,
        "limitations": [
            "assumes FP32 model checkpoint tensors and two FP32 Adam moments",
            "includes 5 percent serialization overhead and max(20 percent,20GiB) free-space headroom",
            "free bytes already reflect retained resume-run files; only future numbered checkpoints are added",
        ],
    }
    fingerprint = canonical_fingerprint(contract)
    report = {**contract, "disk_preflight_fingerprint_sha256": fingerprint}
    report_path = report_root.resolve() / "disk_preflight" / f"disk-preflight-{fingerprint[:16]}.json"
    report_sha256 = write_json_immutable(report_path, report)
    result = {
        **report,
        "report_path": str(report_path),
        "report_sha256": report_sha256,
    }
    if execute and not result["sufficient"]:
        raise RuntimeError(
            "production AE disk preflight failed: "
            f"free={usage.free} required={required_free_bytes} report={report_path}"
        )
    return result


@contextmanager
def _acquire_launch_storage(
    storage: Mapping[str, Any], disk_preflight: Mapping[str, Any]
) -> Iterator[None]:
    # The lease must precede every mutable-root check. Otherwise a compliant
    # active worker can advance the archive after our check, release its lease,
    # and let this process launch from a stale snapshot.
    root = _absolute_path_without_symlink_resolution(Path(str(storage["checkpoint_root"])))
    resume_root = _absolute_path_without_symlink_resolution(Path(str(storage["resume_root"])))
    if storage["launch_mode"] == "fresh":
        try:
            root.mkdir(mode=0o750, parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise RuntimeError(
                f"fresh checkpoint root was claimed after preflight: {root}"
            ) from exc
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError(f"cannot safely open exact checkpoint root for lease: {root}") from exc
    try:
        lock_fd = os.open(
            ".weightclip_ae_launch.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
    except Exception:
        os.close(root_fd)
        raise
    handle = os.fdopen(lock_fd, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another AE launch owns checkpoint root: {root}") from exc
        root = _reject_symlink_components(root, context="checkpoint root under launch lease")
        if (
            os.fstat(root_fd).st_dev != root.stat().st_dev
            or os.fstat(root_fd).st_ino != root.stat().st_ino
        ):
            raise RuntimeError("checkpoint root identity changed while acquiring its launch lease")
        resume_root = _reject_symlink_components(
            resume_root, context="resume root under launch lease"
        )
        if storage["launch_mode"] == "fresh":
            (root / "stage_1").mkdir(mode=0o750, exist_ok=False)
            resume_root.mkdir(mode=0o750, exist_ok=False)
        elif not root.is_dir() or not resume_root.is_dir():
            raise RuntimeError("resume checkpoint/resume roots disappeared before launch")
        if root.stat().st_dev != resume_root.stat().st_dev:
            raise RuntimeError("checkpoint and resume roots must reside on one gated filesystem")
        probe = Path(str(disk_preflight["filesystem_probe_path"]))
        probe_stat = file_stat_identity(probe)
        expected_probe = disk_preflight["filesystem_probe_stat"]
        if (
            probe_stat["device"] != expected_probe["device"]
            or probe_stat["inode"] != expected_probe["inode"]
        ):
            raise RuntimeError("checkpoint filesystem identity changed after disk preflight")
        free_now = shutil.disk_usage(probe).free
        if free_now < int(disk_preflight["required_free_bytes"]):
            raise RuntimeError(
                "checkpoint filesystem free space fell below the gated requirement before launch"
            )
        snapshot = storage.get("resume_state_checkpoint_snapshot")
        if snapshot is not None:
            assert_file_snapshot(snapshot, context="resume-state checkpoint under launch lease")
        archive = storage.get("prior_model_archive")
        if isinstance(archive, Mapping):
            rows = archive.get("artifacts", [])
            if not isinstance(rows, list):
                raise RuntimeError("prior model archive rows are not a list")
            numbered_rows = [row for row in rows if row.get("kind") != "latest_alias"]
            expected_numbered = {Path(str(row.get("path", ""))) for row in numbered_rows}
            actual_numbered = set((root / "stage_1").glob("step_*.pt"))
            if actual_numbered != expected_numbered:
                raise RuntimeError(
                    "prior model archive path inventory changed under launch lease: "
                    f"missing={sorted(str(path) for path in expected_numbered - actual_numbered)[:8]} "
                    f"unexpected={sorted(str(path) for path in actual_numbered - expected_numbered)[:8]}"
                )
            for row in rows:
                path = Path(str(row.get("path", "")))
                assert_file_snapshot(row, context="prior model archive under launch lease")
                if row.get("kind") != "latest_alias" and path.stat().st_mode & 0o222:
                    raise RuntimeError(f"numbered model checkpoint lost immutability: {path}")
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            os.close(root_fd)


@contextmanager
def _acquire_gpu_launch_lease(
    physical_uuid: str,
    *,
    approved_peak_nvml_mib: float,
    lease_dir: Path = Path("/tmp/weightclip_ae_gpu_leases"),
    runner: Any = subprocess.run,
) -> Iterator[dict[str, Any]]:
    """Lease and recheck the exact approved physical GPU immediately before launch."""

    if not physical_uuid.startswith("GPU-"):
        raise RuntimeError("production GPU lease requires a canonical GPU-* UUID")
    lease_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    try:
        lease_dir_fd = os.open(lease_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError(f"production GPU lease directory is unsafe: {lease_dir}") from exc
    lease_dir_stat = os.fstat(lease_dir_fd)
    if lease_dir_stat.st_uid != os.geteuid() or stat.S_IMODE(lease_dir_stat.st_mode) & 0o022:
        os.close(lease_dir_fd)
        raise RuntimeError("production GPU lease directory must be owned by this uid and not group/world writable")
    safe_uuid = physical_uuid.removeprefix("GPU-")
    if not safe_uuid or any(character not in "0123456789abcdefABCDEF-" for character in safe_uuid):
        raise RuntimeError("production GPU UUID contains unsafe lease-path characters")
    try:
        lock_fd = os.open(
            f"{physical_uuid}.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=lease_dir_fd,
        )
    except Exception:
        os.close(lease_dir_fd)
        raise
    lock_stat = os.fstat(lock_fd)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid():
        os.close(lock_fd)
        os.close(lease_dir_fd)
        raise RuntimeError("production GPU lease file must be a regular file owned by this uid")
    handle = os.fdopen(lock_fd, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another bounded/production run owns GPU {physical_uuid}") from exc

        apps = runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        foreign_rows = [
            line.strip()
            for line in str(apps.stdout).splitlines()
            if line.strip() and line.split(",", maxsplit=1)[0].strip() == physical_uuid
        ]
        if foreign_rows:
            raise RuntimeError(
                f"approved GPU {physical_uuid} already has compute applications: {foreign_rows}"
            )
        memory = runner(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                f"--id={physical_uuid}",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        rows = [line.strip() for line in str(memory.stdout).splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(f"approved GPU memory query is ambiguous: {rows}")
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 4 or fields[0] != physical_uuid:
            raise RuntimeError(f"approved GPU memory query returned wrong identity: {rows[0]!r}")
        total_mib, used_mib, free_mib = (float(value) for value in fields[1:])
        if not all(math.isfinite(value) and value >= 0.0 for value in (total_mib, used_mib, free_mib)):
            raise RuntimeError("approved GPU memory query returned invalid capacity values")
        headroom_mib = max(
            PRODUCTION_GPU_HEADROOM_MIN_MIB,
            PRODUCTION_GPU_HEADROOM_FRACTION * total_mib,
        )
        required_free_mib = float(approved_peak_nvml_mib) + headroom_mib
        if free_mib < required_free_mib:
            raise RuntimeError(
                "approved GPU has insufficient free memory: "
                f"free_mib={free_mib} required_mib={required_free_mib} "
                f"profile_peak_mib={approved_peak_nvml_mib} headroom_mib={headroom_mib}"
            )
        yield {
            "physical_gpu_uuid": physical_uuid,
            "total_mib": total_mib,
            "used_mib": used_mib,
            "free_mib": free_mib,
            "approved_peak_nvml_mib": float(approved_peak_nvml_mib),
            "headroom_mib": headroom_mib,
            "required_free_mib": required_free_mib,
            "compute_applications": [],
        }
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            os.close(lease_dir_fd)


def _operator_bank_overrides(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    pair_manifest: Path | None = None,
) -> tuple[list[str], Path]:
    data = cfg.get("data_contract", {})
    if data.get("mode") != "operator_bank":
        raise RuntimeError("Production AE launch requires data_contract.mode=operator_bank")
    configured_pair_value = data.get("pair_manifest")
    if pair_manifest is not None:
        pair_path = pair_manifest.expanduser().resolve()
        if configured_pair_value:
            configured_pair_path = _resolve_path(str(configured_pair_value), relative_to=config_path.parent)
            if configured_pair_path != pair_path:
                raise RuntimeError(
                    "Explicit --pair-manifest disagrees with data_contract.pair_manifest: "
                    f"explicit={pair_path} configured={configured_pair_path}"
                )
        pair_value: str | Path | None = pair_path
    else:
        pair_value = configured_pair_value
    if not pair_value:
        raise RuntimeError(
            "Production AE launch blocked: provide --pair-manifest or set data_contract.pair_manifest"
        )
    pair_path = (
        Path(pair_value).expanduser().resolve()
        if pair_manifest is not None
        else _resolve_path(str(pair_value), relative_to=config_path.parent)
    )
    if not pair_path.is_file():
        raise RuntimeError(f"Production AE launch blocked: operator-bank pair manifest is missing: {pair_path}")
    pair = json.loads(pair_path.read_text())
    protocol = pair.get("contract", {}).get("protocol", {})
    expected_shape = (int(data["tile_rows"]), int(data["tile_cols"]))
    actual_shape = (int(protocol.get("tile_rows", -1)), int(protocol.get("tile_cols", -1)))
    if expected_shape != (128, 128) or actual_shape != expected_shape:
        raise RuntimeError(f"Operator-bank tile contract mismatch: config={expected_shape} bank={actual_shape}")
    if int(data["activation_rows"]) != 512 or int(protocol.get("activation_rows", -1)) != 512:
        raise RuntimeError("Production AE requires exactly 512 activation rows")
    if int(data["permutation_views"]) != 5 or int(protocol.get("permutation_views", -1)) != 5:
        raise RuntimeError("Production AE requires exactly five graph-gauge views")
    if abs(float(data["canonical_probability"]) - 1.0 / 6.0) > 1e-12:
        raise RuntimeError("Official WeightCLIP mixture requires canonical_probability=1/6")
    patch_size = int(data["model_patch_size"])
    if patch_size <= 0 or 128 % patch_size:
        raise RuntimeError("model_patch_size must divide the fixed 128-row operator tile")
    loss = data["loss_recipe"]
    pair_sha256 = sha256_file(pair_path)
    stream_seed = int(data.get("seed", 42))
    locality_source = OperatorBankTrainingDataset(
        pair_path,
        seed=stream_seed,
        repeat=True,
        permutation_views=True,
        canonical_probability=float(data["canonical_probability"]),
        hot_shards=int(data["hot_shards"]),
        expected_pair_manifest_sha256=pair_sha256,
    )
    locality_audit = locality_source.locality_audit(0)
    locality_bundle_contract = OperatorBankBundleDataset(
        locality_source,
        max_active_strata=int(data["max_active_strata"]),
        max_active_bundle_bytes=int(data["max_active_bundle_bytes"]),
    )
    loader_workers = int(data["loader_workers"])
    inflight_bundle_bound = int(data["loader_batch_size"]) * (
        (loader_workers * int(data["loader_prefetch_factor"]) + 1) if loader_workers > 0 else 1
    )
    inflight_tensor_bytes_bound = inflight_bundle_bound * locality_bundle_contract.max_bundle_tensor_bytes
    locality_audit.update(
        {
            "pair_manifest_sha256": pair_sha256,
            "effective_seed": stream_seed,
            "rank": 0,
            "world_size": 1,
            "temporal_order_deviation": (
                "lineage/checkpoint order may change within each frozen dataset/op/depth/role stratum; "
                "the global stratum sequence and exact entry multiset are preserved"
            ),
            "memory_contract": {
                "active_strata_tensor_bytes_bound": locality_bundle_contract.active_tensor_bytes_bound,
                "configured_max_active_tensor_bytes": int(data["max_active_bundle_bytes"]),
                "max_single_bundle_tensor_bytes": locality_bundle_contract.max_bundle_tensor_bytes,
                "inflight_bundle_count_bound": inflight_bundle_bound,
                "inflight_tensor_bytes_bound": inflight_tensor_bytes_bound,
                "configured_max_inflight_bundle_count": int(data["max_inflight_bundles"]),
                "configured_max_inflight_tensor_bytes": int(data["max_inflight_bundle_bytes"]),
                "active_plus_inflight_tensor_bytes_bound": (
                    locality_bundle_contract.active_tensor_bytes_bound + inflight_tensor_bytes_bound
                ),
            },
        }
    )
    locality_audit_path = pair_path.parent / (
        f"operator_bank_locality_audit_{pair_sha256[:16]}_seed{stream_seed}.json"
    )
    locality_audit_sha256 = write_json_immutable(locality_audit_path, locality_audit)
    values = {
        "+train.operator_bank.enabled": True,
        "+train.operator_bank.approved_weightclip_contract": True,
        "+train.operator_bank.pair_manifest": str(pair_path),
        "+train.operator_bank.pair_manifest_sha256": pair_sha256,
        "+train.operator_bank.locality_audit_path": str(locality_audit_path.resolve()),
        "+train.operator_bank.locality_audit_sha256": locality_audit_sha256,
        "+train.operator_bank.repeat": True,
        "+train.operator_bank.permutation_views": True,
        "+train.operator_bank.canonical_probability": float(data["canonical_probability"]),
        "+train.operator_bank.hot_shards": int(data["hot_shards"]),
        "+train.operator_bank.shard_by_rank": True,
        "+train.operator_bank.loader_workers": int(data["loader_workers"]),
        "+train.operator_bank.loader_batch_size": int(data["loader_batch_size"]),
        "+train.operator_bank.loader_prefetch_factor": int(data["loader_prefetch_factor"]),
        "+train.operator_bank.loader_persistent_workers": True,
        "+train.operator_bank.max_active_strata": int(data["max_active_strata"]),
        "+train.operator_bank.max_active_bundle_bytes": int(data["max_active_bundle_bytes"]),
        "+train.operator_bank.max_inflight_bundles": int(data["max_inflight_bundles"]),
        "+train.operator_bank.max_inflight_bundle_bytes": int(data["max_inflight_bundle_bytes"]),
        "train.device": "cuda:0",
        "train.distributed": False,
        "train.num_gpus": 1,
        "data.seed": stream_seed,
        "train.offline_dataset.enabled": False,
        "train.preslicing.enabled": False,
        "train.synthetic_layer_source.enabled": False,
        "train.max_x_rows": 512,
        "train.stage": 1,
        "train.stage_base_T_patches": 128 // patch_size,
        "train.stage_base_d_out": 128,
        "train.slice_batch_size": int(data["slice_batch_size"]),
        "model.patch_size": patch_size,
        "model.big_vae.distribution_encoder.patch_size_for_cov": patch_size,
        "train.behavioral_coef": float(loss["behavioral_coef"]),
        "train.behavioral_loss.lambda_operator": float(loss["behavioral_lambda_operator"]),
        "train.behavioral_loss.lambda_dir": float(loss["behavioral_lambda_dir"]),
        "train.behavioral_loss.lambda_scale": float(loss["behavioral_lambda_scale"]),
        "train.structural_coef": float(loss["structural_coef"]),
        "train.struct_loss.lambda_dir": float(loss["structural_lambda_dir"]),
        "train.struct_loss.lambda_scale": float(loss["structural_lambda_scale"]),
        "train.struct_loss.lambda_rec": float(loss["structural_lambda_rec"]),
        "train.struct_loss.lambda_rel": float(loss["structural_lambda_rel"]),
        "train.resume_state.enabled": True,
        "train.resume_state.auto_resume": False,
        "train.resume_state.load_model_state": True,
        "train.resume_state.load_optimizer_state": True,
        "train.resume_state.load_scheduler_state": True,
        "train.resume_state.load_scaler_state": True,
        "train.resume_state.load_rng_state": True,
        "train.resume_state.load_step": True,
        # The historical every-five-step wide CSV grew to tens of GB. Keep
        # periodic top-k telemetry in logs without recreating that artifact.
        "train.telemetry.grad_layer_monitor.enabled": True,
        "train.telemetry.grad_layer_monitor.every_steps": 100,
        "train.telemetry.grad_layer_monitor.save_csv": False,
        "train.telemetry.grad_layer_monitor.save_plot": False,
        "train.telemetry.grad_layer_monitor.save_heatmap": False,
    }
    overrides = [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in values.items()]
    return overrides, pair_path


def _assert_resolved_contract(resolved: dict[str, Any], pair_path: Path) -> None:
    train = resolved["train"]
    model = resolved["model"]
    bank = train["operator_bank"]
    expected = {
        "device": "cuda:0",
        "distributed": False,
        "num_gpus": 1,
        "max_x_rows": 512,
        "slice_batch_size": 32,
        "offline_enabled": False,
        "preslicing_enabled": False,
        "synthetic_enabled": False,
        "operator_enabled": True,
        "max_active_strata": 256,
        "max_active_bundle_bytes": 1610612736,
        "loader_batch_size": 1,
        "loader_prefetch_factor": 2,
        "max_inflight_bundles": 17,
        "max_inflight_bundle_bytes": 134217728,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "locality_audit_path": bank["locality_audit_path"],
        "locality_audit_sha256": bank["locality_audit_sha256"],
        "patch_size": 16,
        "behavioral_coef": 1.0,
        "behavioral_operator": 50.0,
        "behavioral_dir": 1.0,
        "behavioral_scale": 10.0,
        "structural_coef": 1.0,
        "structural_dir": 1.0,
        "structural_scale": 10.0,
        "structural_rec": 0.0,
        "structural_rel": 0.0,
    }
    actual = {
        "device": train["device"],
        "distributed": train["distributed"],
        "num_gpus": train["num_gpus"],
        "max_x_rows": train["max_x_rows"],
        "slice_batch_size": train["slice_batch_size"],
        "offline_enabled": train["offline_dataset"]["enabled"],
        "preslicing_enabled": train["preslicing"]["enabled"],
        "synthetic_enabled": train["synthetic_layer_source"]["enabled"],
        "operator_enabled": bank["enabled"],
        "max_active_strata": bank["max_active_strata"],
        "max_active_bundle_bytes": bank["max_active_bundle_bytes"],
        "loader_batch_size": bank["loader_batch_size"],
        "loader_prefetch_factor": bank["loader_prefetch_factor"],
        "max_inflight_bundles": bank["max_inflight_bundles"],
        "max_inflight_bundle_bytes": bank["max_inflight_bundle_bytes"],
        "pair_manifest": bank["pair_manifest"],
        "pair_manifest_sha256": bank["pair_manifest_sha256"],
        "locality_audit_path": bank["locality_audit_path"],
        "locality_audit_sha256": bank["locality_audit_sha256"],
        "patch_size": model["patch_size"],
        "behavioral_coef": train["behavioral_coef"],
        "behavioral_operator": train["behavioral_loss"]["lambda_operator"],
        "behavioral_dir": train["behavioral_loss"]["lambda_dir"],
        "behavioral_scale": train["behavioral_loss"]["lambda_scale"],
        "structural_coef": train["structural_coef"],
        "structural_dir": train["struct_loss"]["lambda_dir"],
        "structural_scale": train["struct_loss"]["lambda_scale"],
        "structural_rec": train["struct_loss"]["lambda_rec"],
        "structural_rel": train["struct_loss"]["lambda_rel"],
    }
    if actual != expected:
        mismatches = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}
        raise RuntimeError(f"Resolved production contract mismatch: {mismatches}")
    locality_audit_path = Path(str(bank["locality_audit_path"]))
    if not locality_audit_path.is_file() or sha256_file(locality_audit_path) != str(bank["locality_audit_sha256"]):
        raise RuntimeError("Resolved production locality audit is missing or not SHA-bound")
    locality_audit = json.loads(locality_audit_path.read_text(encoding="utf-8"))
    if (
        locality_audit.get("pair_manifest_sha256") != sha256_file(pair_path)
        or not bool(locality_audit.get("exact_entry_multiset", False))
        or not bool(locality_audit.get("exact_global_stratum_sequence", False))
        or not bool(locality_audit.get("lineage_checkpoint_temporal_order_is_not_preserved_by_contract", False))
    ):
        raise RuntimeError("Resolved production locality audit violates the frozen order contract")
    resume = train["resume_state"]
    if not all(bool(resume[key]) for key in (
        "enabled", "load_model_state", "load_optimizer_state",
        "load_scheduler_state", "load_scaler_state", "load_step",
        "load_rng_state",
    )):
        raise RuntimeError(f"Resolved production resume contract is incomplete: {resume}")
    if bool(resume["auto_resume"]):
        raise RuntimeError("Production AE forbids directory-scanning auto-resume")
    monitor = train["telemetry"]["grad_layer_monitor"]
    if bool(monitor["save_csv"]) or int(monitor["every_steps"]) < 100:
        raise RuntimeError(f"Resolved telemetry would recreate high-volume per-layer CSV logging: {monitor}")


def _validate_candidate_artifact_index(
    path: Path,
    *,
    candidate_name: str,
    model_fingerprint_sha256: str,
    trainable_parameters: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"candidate artifact index is missing: {path}")
    index = json.loads(path.read_text(encoding="utf-8"))
    artifact_set = str(index.get("artifact_set_fingerprint_sha256", ""))
    if len(artifact_set) != 64 or path.parent.name != f"candidate_profiles-{artifact_set[:16]}":
        raise RuntimeError("candidate artifact index is not content-addressed by its artifact-set fingerprint")
    report_ref = index.get("report", {})
    report_path = Path(str(report_ref.get("path", ""))).resolve()
    report_sha256 = str(report_ref.get("sha256", ""))
    if not report_path.is_file() or sha256_file(report_path) != report_sha256:
        raise RuntimeError("candidate profile report is missing, tampered, or not SHA-bound")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_contract = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "artifact_set_fingerprint_sha256",
            "config_path",
            "base_model_config_path",
        }
    }
    recomputed_artifact_set = canonical_fingerprint(report_contract)
    if recomputed_artifact_set != artifact_set:
        raise RuntimeError(
            "candidate artifact-set fingerprint does not match the canonical report content: "
            f"declared={artifact_set} recomputed={recomputed_artifact_set}"
        )
    source_ref = index.get("source_implementation", {})
    source_path = Path(str(source_ref.get("path", ""))).resolve()
    if not source_path.is_file() or sha256_file(source_path) != source_ref.get("sha256"):
        raise RuntimeError("candidate source implementation seal artifact is missing or tampered")
    source_seal = json.loads(source_path.read_text(encoding="utf-8"))
    source_seal_sha256 = validate_ae_source_implementation_seal(source_seal)
    if (
        report.get("artifact_set_fingerprint_sha256") != artifact_set
        or report.get("fingerprint_source") != "exact_hydra_composed_worker_model"
        or report.get("source_implementation_seal_sha256") != source_seal_sha256
        or source_ref.get("source_implementation_seal_sha256") != source_seal_sha256
    ):
        raise RuntimeError("candidate profile report disagrees with its artifact index")
    profiles = [
        row for row in report.get("profiles", [])
        if row.get("candidate", {}).get("name") == candidate_name
    ]
    if len(profiles) != 1:
        raise RuntimeError(f"candidate profile report does not contain exactly one {candidate_name!r} row")
    profile = profiles[0]
    if (
        profile.get("model_fingerprint_sha256") != model_fingerprint_sha256
        or profile.get("resolved_model_config_sha256") != model_fingerprint_sha256
        or int(profile.get("parameter_ledger", {}).get("trainable_parameters", -1)) != trainable_parameters
    ):
        raise RuntimeError("candidate profile fingerprint/parameter ledger disagrees with production resolution")
    return {
        "artifact_set_fingerprint_sha256": artifact_set,
        "candidate_report_path": str(report_path),
        "candidate_report_sha256": report_sha256,
        "candidate_index_path": str(path.resolve()),
        "candidate_index_sha256": sha256_file(path),
        "source_implementation_seal_path": str(source_path),
        "source_implementation_seal_artifact_sha256": sha256_file(source_path),
        "source_implementation_seal_sha256": source_seal_sha256,
    }


def _validate_runtime_profile_summary(
    path: Path,
    *,
    candidate_name: str,
    model_fingerprint_sha256: str,
    trainable_parameters: int,
    pair_manifest_sha256: str,
    production_scientific_sha256: str,
    source_implementation_seal_sha256: str,
    candidate_artifact_set_sha256: str,
    candidate_report_sha256: str,
    candidate_index_sha256: str,
    production_resolved: dict[str, Any],
    expected_profile_mode: str = "production_exact",
    require_launcher_evidence: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"successful bounded runtime profile summary is missing: {path}")
    summary_sha256 = sha256_file(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    actual_model = summary.get("actual_model", {})
    actual_model_config = actual_model.get("config")
    gpu_identity = summary.get("gpu_identity", {})
    peak_vram = summary.get("peak_vram_mib", {})
    refill = summary.get("measured_producer_refill", {})
    producer_close = summary.get("producer_close", {})
    stress_gate = summary.get("producer_stress_gate", {})
    expected_horizon_steps = 84 if expected_profile_mode == "production_exact" else 64
    if expected_profile_mode not in {"production_exact", "producer_stress"}:
        raise RuntimeError(f"unknown expected bounded profile mode: {expected_profile_mode!r}")
    if (
        summary.get("profile_schema") != "weightclip_ae_runtime_v2"
        or summary.get("profile_mode") != expected_profile_mode
        or summary.get("termination_reason") != "bounded_profile_complete"
        or summary.get("candidate") != candidate_name
        or summary.get("model_fingerprint_sha256") != model_fingerprint_sha256
        or summary.get("pair_manifest_sha256") != pair_manifest_sha256
        or summary.get("source_implementation_seal_sha256") != source_implementation_seal_sha256
        or summary.get("candidate_artifact_set_sha256") != candidate_artifact_set_sha256
        or summary.get("candidate_report_sha256") != candidate_report_sha256
        or summary.get("candidate_index_sha256") != candidate_index_sha256
        or actual_model.get("config_sha256") != model_fingerprint_sha256
        or not isinstance(actual_model_config, dict)
        or canonical_fingerprint(actual_model_config) != model_fingerprint_sha256
        or int(actual_model.get("trainable_parameters", -1)) != trainable_parameters
        or not str(gpu_identity.get("physical_uuid", "")).startswith("GPU-")
        or gpu_identity.get("logical_device") != "cuda:0"
        or not math.isfinite(float(peak_vram.get("nvml_used", float("nan"))))
        or float(peak_vram.get("nvml_used", -1.0)) <= 0.0
        or summary.get("input_validation", {}).get("status") != "passed"
        or bool(summary.get("external_tracking_enabled", True))
        or bool(summary.get("checkpoint_files_found", ["missing"]))
        or int(summary.get("warmup_steps", -1)) != 4
        or int(summary.get("measured_steps", -1)) != 28
        or int(summary.get("executed_optimizer_steps", -1)) != 32
        or refill.get("required_by_frozen_protocol") is not True
        or refill.get("observed") is not True
        or refill.get("monotonic_nondecreasing") is not True
        or int(refill.get("advance_tiles", 0)) <= 0
        or producer_close.get("unconsumed_tail_required") is not True
        or producer_close.get("unconsumed_tail_observed") is not True
        or int(producer_close.get("unconsumed_discarded_tiles", 0)) <= 0
        or producer_close.get("unconsumed_tail_commit_policy") != "discard_without_commit"
        or producer_close.get("profile_mode") != expected_profile_mode
        or int(producer_close.get("requested_horizon_steps", -1)) != expected_horizon_steps
        or int(producer_close.get("requested_horizon_tiles", -1)) != expected_horizon_steps * 32
        or int(producer_close.get("tiles_per_optimizer_step", -1)) != 32
        or producer_close.get("horizon_units")
        != {
            "requested_horizon_steps": "optimizer_steps",
            "requested_horizon_tiles": "logical_tiles",
            "tiles_per_optimizer_step": "logical_tiles_per_optimizer_step",
        }
    ):
        raise RuntimeError("bounded runtime profile summary violates candidate/data/success bindings")
    if expected_profile_mode == "producer_stress" and (
        stress_gate.get("applicable") is not True
        or stress_gate.get("status") != "passed"
        or int(refill.get("advance_tiles", -1)) < 832
        or float(refill.get("producer_to_consumer_span_ratio", -1.0)) < 26.0 / 27.0
        or float(summary.get("input_wait_ms", {}).get("fraction_of_measured_wall", 1.0)) > 0.01
        or float(summary.get("input_wait_ms", {}).get("p95", float("inf")))
        > 0.05 * float(summary.get("host_step_ms", {}).get("median", 0.0))
    ):
        raise RuntimeError("producer_stress summary violates frozen ingress thresholds")
    if expected_profile_mode == "production_exact" and stress_gate.get("applicable") is not False:
        raise RuntimeError("production_exact summary must mark producer_stress gate not applicable")
    contract_path = Path(str(summary.get("profile_contract_path", ""))).resolve()
    contract_sha256 = str(summary.get("profile_contract_sha256", ""))
    if not contract_path.is_file() or sha256_file(contract_path) != contract_sha256:
        raise RuntimeError("bounded runtime profile contract is missing or tampered")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract = {
        "profile_schema": "weightclip_ae_runtime_v2",
        "profile_mode": expected_profile_mode,
        "prefetch_queue_size": 24 if expected_profile_mode == "production_exact" else 4,
        "candidate": candidate_name,
        "model_fingerprint_sha256": model_fingerprint_sha256,
        "pair_manifest_sha256": pair_manifest_sha256,
        "production_scientific_config_sha256": production_scientific_sha256,
        "source_implementation_seal_sha256": source_implementation_seal_sha256,
        "candidate_artifact_set_sha256": candidate_artifact_set_sha256,
        "candidate_report_sha256": candidate_report_sha256,
        "candidate_index_sha256": candidate_index_sha256,
        "resolved_profile_config": str(Path(str(summary.get("resolved_config_path", ""))).resolve()),
        "resolved_profile_config_sha256": str(summary.get("resolved_config_sha256", "")),
        "production_max_steps": 500_000,
        "hard_cap_optimizer_steps": 32,
        "warmup_steps": 4,
        "measured_steps": 28,
        "external_tracking": False,
        "resume": False,
        "checkpoint_writes": False,
        "physical_gpu_uuid": str(gpu_identity["physical_uuid"]),
        "cuda_visible_devices": str(gpu_identity["physical_uuid"]),
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"bounded runtime profile contract binding mismatch: {mismatches}")
    if (
        gpu_identity.get("cuda_visible_devices") != gpu_identity.get("physical_uuid")
        or contract.get("physical_gpu_uuid") != gpu_identity.get("physical_uuid")
    ):
        raise RuntimeError("bounded runtime profile did not run under its exact physical-GPU UUID mask")
    if (
        contract.get("production_scientific_config_sha256") != production_scientific_sha256
        or contract.get("source_implementation_seal_sha256") != source_implementation_seal_sha256
    ):
        raise RuntimeError("bounded runtime profile scientific config differs from production")
    source_path = Path(str(contract.get("source_implementation_seal_path", ""))).resolve()
    if (
        not source_path.is_file()
        or sha256_file(source_path) != contract.get("source_implementation_seal_artifact_sha256")
    ):
        raise RuntimeError("bounded runtime profile source implementation artifact is missing or tampered")
    runtime_source = json.loads(source_path.read_text(encoding="utf-8"))
    if validate_ae_source_implementation_seal(runtime_source) != source_implementation_seal_sha256:
        raise RuntimeError("bounded runtime profile source implementation differs from production")
    resolved_path = Path(str(summary.get("resolved_config_path", ""))).resolve()
    if not resolved_path.is_file() or sha256_file(resolved_path) != summary.get("resolved_config_sha256"):
        raise RuntimeError("bounded runtime profile resolved config is missing or tampered")
    profile_resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    normalized_profile = copy.deepcopy(profile_resolved)
    profile_queue = int(normalized_profile["train"]["offline_batch_prefetch"]["queue_size"])
    expected_queue = 24 if expected_profile_mode == "production_exact" else 4
    if profile_queue != expected_queue:
        raise RuntimeError(
            f"bounded runtime profile resolved queue mismatch: mode={expected_profile_mode} "
            f"expected={expected_queue} actual={profile_queue}"
        )
    normalized_profile["train"]["offline_batch_prefetch"]["queue_size"] = 24
    assert_bounded_profile_production_parity(production_resolved, normalized_profile)
    launcher_result_path = path.parent / "launcher_result.json"
    if not require_launcher_evidence:
        return {
            "runtime_profile_summary_path": str(path.resolve()),
            "runtime_profile_summary_sha256": summary_sha256,
            "runtime_profile_contract_path": str(contract_path),
            "runtime_profile_contract_sha256": contract_sha256,
            "runtime_profile_source_implementation_path": str(source_path),
            "runtime_profile_source_implementation_artifact_sha256": sha256_file(source_path),
            "runtime_profile_resolved_config_path": str(resolved_path),
            "runtime_profile_resolved_config_sha256": str(summary["resolved_config_sha256"]),
            "runtime_profile_mode": expected_profile_mode,
            "physical_gpu_uuid": str(gpu_identity["physical_uuid"]),
            "peak_nvml_used_mib": float(peak_vram["nvml_used"]),
        }
    if not launcher_result_path.is_file():
        raise RuntimeError("bounded runtime profile launcher_result.json is missing")
    launcher_result_sha256 = sha256_file(launcher_result_path)
    launcher_result = json.loads(launcher_result_path.read_text(encoding="utf-8"))
    log_ref = launcher_result.get("worker_console_log", {})
    log_path = Path(str(log_ref.get("path", ""))).resolve()
    if (
        launcher_result.get("status") != "validated_success"
        or launcher_result.get("profile_schema") != "weightclip_ae_runtime_v2"
        or launcher_result.get("profile_mode") != expected_profile_mode
        or launcher_result.get("candidate") != candidate_name
        or launcher_result.get("model_fingerprint_sha256") != model_fingerprint_sha256
        or launcher_result.get("pair_manifest_sha256") != pair_manifest_sha256
        or launcher_result.get("source_implementation_seal_sha256")
        != source_implementation_seal_sha256
        or launcher_result.get("candidate_artifact_set_sha256")
        != candidate_artifact_set_sha256
        or launcher_result.get("candidate_index_sha256") != candidate_index_sha256
        or launcher_result.get("summary", {}).get("path") != str(path.resolve())
        or launcher_result.get("summary", {}).get("sha256") != summary_sha256
        or not log_path.is_file()
        or sha256_file(log_path) != log_ref.get("sha256")
    ):
        raise RuntimeError("bounded runtime profile launcher/log evidence is missing or tampered")
    return {
        "runtime_profile_summary_path": str(path.resolve()),
        "runtime_profile_summary_sha256": summary_sha256,
        "runtime_profile_contract_path": str(contract_path),
        "runtime_profile_contract_sha256": contract_sha256,
        "runtime_profile_source_implementation_path": str(source_path),
        "runtime_profile_source_implementation_artifact_sha256": sha256_file(source_path),
        "runtime_profile_resolved_config_path": str(resolved_path),
        "runtime_profile_resolved_config_sha256": str(summary["resolved_config_sha256"]),
        "runtime_profile_launcher_result_path": str(launcher_result_path.resolve()),
        "runtime_profile_launcher_result_sha256": launcher_result_sha256,
        "runtime_profile_worker_log_path": str(log_path),
        "runtime_profile_worker_log_sha256": str(log_ref["sha256"]),
        "runtime_profile_mode": expected_profile_mode,
        "physical_gpu_uuid": str(gpu_identity["physical_uuid"]),
        "peak_nvml_used_mib": float(peak_vram["nvml_used"]),
    }


def _assert_profile_gpu_parity(
    production_exact: Mapping[str, Any], producer_stress: Mapping[str, Any]
) -> tuple[str, float]:
    approved_gpu_uuid = str(production_exact.get("physical_gpu_uuid", ""))
    stress_gpu_uuid = str(producer_stress.get("physical_gpu_uuid", ""))
    if not approved_gpu_uuid.startswith("GPU-") or stress_gpu_uuid != approved_gpu_uuid:
        raise RuntimeError(
            "production_exact and producer_stress profiles used different physical GPUs: "
            f"production_exact={approved_gpu_uuid} producer_stress={stress_gpu_uuid}"
        )
    peak_nvml_mib = float(production_exact.get("peak_nvml_used_mib", float("nan")))
    if not math.isfinite(peak_nvml_mib) or peak_nvml_mib <= 0.0:
        raise RuntimeError("production_exact profile has no valid NVML peak-memory evidence")
    return approved_gpu_uuid, peak_nvml_mib


def _production_child_environment(
    physical_gpu_uuid: str, *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    if not physical_gpu_uuid.startswith("GPU-"):
        raise RuntimeError("production child environment requires a canonical GPU-* UUID")
    child_environment = dict(os.environ if environ is None else environ)
    child_environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_uuid
    return child_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approval-gated WeightCLIP production AE launcher")
    parser.add_argument("--config", type=Path, default=Path("conf/weightclip_benchmark/ae_700m.yaml"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--approval-request", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--candidate-artifact-index", type=Path, required=True)
    parser.add_argument("--runtime-profile-summary", type=Path, required=True)
    parser.add_argument("--producer-stress-summary", type=Path, required=True)
    parser.add_argument("--launch-mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--resume-state-checkpoint", type=Path)
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        required=True,
        help="Exact finalized operator-bank pair manifest; canonical ae_700m.yaml remains pending/null",
    )
    parser.add_argument("--execute", action="store_true", help="Execute after approval; default is a verified dry run")
    parser.add_argument("hydra_overrides", nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    cfg = _load_yaml(config_path)
    from training.weightclip_benchmark.approve_ae import validate_approved_transition

    transition = validate_approved_transition(
        approval_request_path=args.approval_request.resolve(),
        approved_config_path=config_path,
        approval_path=args.approval_file.resolve(),
    )
    if transition["approved_config_path"] != config_path:
        raise RuntimeError("--config is not the approved config bound by the approval transition")
    configured_approval = _resolve_path(
        str(cfg.get("approval_file", "")), relative_to=config_path.parent
    )
    if cfg.get("selected_candidate") != args.candidate:
        raise RuntimeError("--candidate differs from approved config selected_candidate")
    if configured_approval != args.approval_file.resolve():
        raise RuntimeError("--approval-file differs from approved config approval_file")
    request = transition["request"]
    exact_cli_inputs = {
        "candidate_artifact_index": args.candidate_artifact_index.resolve(),
        "pair_manifest": args.pair_manifest.resolve(),
        "production_exact_summary": args.runtime_profile_summary.resolve(),
        "producer_stress_summary": args.producer_stress_summary.resolve(),
    }
    for name, actual_path in exact_cli_inputs.items():
        expected_path = Path(request["inputs"][name]["path"]).resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                f"--{name.replace('_', '-')} differs from the approved request: "
                f"actual={actual_path} expected={expected_path}"
            )
    if not bool(cfg.get("production_launch_enabled_after_approval", False)):
        raise RuntimeError("Production AE launch blocked by config: production_launch_enabled_after_approval=false")
    axes = _candidate(cfg, args.candidate)
    base_path = _resolve_path(str(cfg["base_model_config"]), relative_to=config_path.parent)
    candidate = resolve_ae_candidate(apply_scaling_axes(_load_yaml(base_path), axes))
    fingerprint = candidate.model_fingerprint_sha256
    tile_rows, tile_cols = map(int, cfg["tile_shape"])
    print(
        "[weightclip-ae-train] stage=approval-preflight "
        f"candidate={axes.name} fingerprint={fingerprint} config={config_path} "
        f"device=hydra-resolved dtype=amp-resolved seed=hydra-resolved cache_mode=offline-dataset",
        flush=True,
    )
    validate_production_operational_overrides(args.hydra_overrides)
    storage = _launch_storage_contract(
        launch_mode=args.launch_mode,
        checkpoint_root=args.checkpoint_root,
        resume_state_checkpoint=args.resume_state_checkpoint,
    )
    bank_overrides, pair_path = _operator_bank_overrides(
        cfg,
        config_path,
        pair_manifest=args.pair_manifest,
    )
    base_production_overrides = [
        *candidate.hydra_model_overrides,
        f"train.max_steps={int(cfg['training_steps'])}",
        "train.kl_beta=0.0",
        "train.kl_schedule.enabled=false",
        *bank_overrides,
        *_storage_overrides(storage),
        *args.hydra_overrides,
    ]
    production_resolved = _compose_resolved(base_production_overrides)
    assert_exact_candidate_model(candidate, production_resolved, context="canonical production launch")
    _assert_resolved_contract(production_resolved, pair_path)
    parameter_ledger = exact_parameter_ledger(candidate.resolved_model_config)
    trainable_parameters = int(parameter_ledger["trainable_parameters"])
    target_parameters = int(cfg["target_trainable_parameters"])
    relative_error = abs(trainable_parameters - target_parameters) / max(target_parameters, 1)
    if relative_error > float(cfg["target_relative_tolerance"]):
        raise RuntimeError(
            f"Production AE candidate trainable count is outside the approved budget: "
            f"actual={trainable_parameters} target={target_parameters} relative_error={relative_error}"
        )
    scientific_config_sha256 = production_scientific_config_sha256(production_resolved)
    candidate_artifacts = _validate_candidate_artifact_index(
        args.candidate_artifact_index.resolve(),
        candidate_name=axes.name,
        model_fingerprint_sha256=fingerprint,
        trainable_parameters=trainable_parameters,
    )
    pair_manifest_sha256 = sha256_file(pair_path)
    runtime_profile = _validate_runtime_profile_summary(
        args.runtime_profile_summary.resolve(),
        candidate_name=axes.name,
        model_fingerprint_sha256=fingerprint,
        trainable_parameters=trainable_parameters,
        pair_manifest_sha256=pair_manifest_sha256,
        production_scientific_sha256=scientific_config_sha256,
        source_implementation_seal_sha256=str(
            candidate_artifacts["source_implementation_seal_sha256"]
        ),
        candidate_artifact_set_sha256=str(
            candidate_artifacts["artifact_set_fingerprint_sha256"]
        ),
        candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
        candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
        production_resolved=production_resolved,
        expected_profile_mode="production_exact",
    )
    producer_stress_profile = _validate_runtime_profile_summary(
        args.producer_stress_summary.resolve(),
        candidate_name=axes.name,
        model_fingerprint_sha256=fingerprint,
        trainable_parameters=trainable_parameters,
        pair_manifest_sha256=pair_manifest_sha256,
        production_scientific_sha256=scientific_config_sha256,
        source_implementation_seal_sha256=str(
            candidate_artifacts["source_implementation_seal_sha256"]
        ),
        candidate_artifact_set_sha256=str(
            candidate_artifacts["artifact_set_fingerprint_sha256"]
        ),
        candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
        candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
        production_resolved=production_resolved,
        expected_profile_mode="producer_stress",
    )
    approved_gpu_uuid, approved_peak_nvml_mib = _assert_profile_gpu_parity(
        runtime_profile, producer_stress_profile
    )
    _bind_parent_gpu_mask(approved_gpu_uuid)
    requirements = ApprovalRequirements(
        candidate_name=axes.name,
        model_fingerprint_sha256=fingerprint,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        training_steps=int(cfg["training_steps"]),
        production_scientific_config_sha256=scientific_config_sha256,
        candidate_artifact_set_sha256=str(candidate_artifacts["artifact_set_fingerprint_sha256"]),
        candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
        candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
        trainable_parameters=trainable_parameters,
        runtime_profile_summary_sha256=str(runtime_profile["runtime_profile_summary_sha256"]),
        producer_stress_profile_summary_sha256=str(
            producer_stress_profile["runtime_profile_summary_sha256"]
        ),
        pair_manifest_sha256=pair_manifest_sha256,
        source_implementation_seal_sha256=str(
            candidate_artifacts["source_implementation_seal_sha256"]
        ),
        approval_request_sha256=sha256_file(args.approval_request.resolve()),
        approval_request_fingerprint_sha256=str(
            request["approval_request_fingerprint_sha256"]
        ),
        canonical_config_sha256=str(transition["canonical_config_sha256"]),
        approved_config_sha256=str(transition["approved_config_sha256"]),
    )
    approval = require_production_approval(args.approval_file.resolve(), requirements)
    resolved_dir = _resolve_path(str(cfg["profile_output_dir"]), relative_to=config_path.parent)
    launch_preflight = {
        "candidate": axes.name,
        "model_fingerprint_sha256": fingerprint,
        "candidate_config_sha256": sha256_file(config_path),
        "canonical_config_sha256": transition["canonical_config_sha256"],
        "approved_config_sha256": transition["approved_config_sha256"],
        "approval_request_sha256": sha256_file(args.approval_request.resolve()),
        "approval_request_fingerprint_sha256": request[
            "approval_request_fingerprint_sha256"
        ],
        "approval_file_sha256": transition["approval_sha256"],
        "base_model_config_sha256": sha256_file(base_path),
        "resolved_model_config_sha256": fingerprint,
        "production_scientific_config_sha256": scientific_config_sha256,
        "trainable_parameters": trainable_parameters,
        "pair_manifest_sha256": pair_manifest_sha256,
        "artifact_set_fingerprint_sha256": candidate_artifacts[
            "artifact_set_fingerprint_sha256"
        ],
        "candidate_report_sha256": candidate_artifacts["candidate_report_sha256"],
        "candidate_index_sha256": candidate_artifacts["candidate_index_sha256"],
        "runtime_profile_summary_sha256": runtime_profile["runtime_profile_summary_sha256"],
        "producer_stress_profile_summary_sha256": producer_stress_profile[
            "runtime_profile_summary_sha256"
        ],
        "source_implementation_seal_sha256": candidate_artifacts[
            "source_implementation_seal_sha256"
        ],
        "physical_gpu_uuid": approved_gpu_uuid,
        "cuda_visible_devices": approved_gpu_uuid,
        "approved_profile_peak_nvml_mib": approved_peak_nvml_mib,
        "gpu_headroom_fraction": PRODUCTION_GPU_HEADROOM_FRACTION,
        "gpu_headroom_min_mib": PRODUCTION_GPU_HEADROOM_MIN_MIB,
    }
    storage = _validate_resume_state_checkpoint(
        storage,
        expected_preflight=launch_preflight,
        expected_scientific_config_sha256=scientific_config_sha256,
        resolved_launch=production_resolved,
        resolved_model_config=candidate.resolved_model_config,
    )
    disk_preflight = _production_disk_preflight(
        resolved=production_resolved,
        parameter_ledger=parameter_ledger,
        report_root=resolved_dir,
        candidate_name=axes.name,
        approved_config_sha256=str(transition["approved_config_sha256"]),
        storage=storage,
        execute=bool(args.execute),
    )
    launch_preflight.update(
        {
            "launch_mode": storage["launch_mode"],
            "checkpoint_root": storage["checkpoint_root"],
            "resume_root": storage["resume_root"],
            "resume_state_checkpoint": storage["resume_state_checkpoint"] or "none",
            "resume_state_checkpoint_sha256": storage["resume_state_checkpoint_sha256"]
            or "none",
            "resume_step": storage["resume_step"],
            "prior_model_archive_fingerprint_sha256": storage.get(
                "prior_model_archive_fingerprint_sha256", "none"
            ),
            "prior_model_archive_artifact_count": storage.get(
                "prior_model_archive_artifact_count", 0
            ),
            "disk_preflight_report_sha256": disk_preflight["report_sha256"],
            "disk_preflight_report_path": disk_preflight["report_path"],
            "disk_preflight_required_free_bytes": disk_preflight["required_free_bytes"],
            "disk_preflight_observed_free_bytes": disk_preflight["observed_free_bytes"],
        }
    )
    preflight_overrides = [
        f"+weightclip_launch_preflight.{key}={value}" for key, value in launch_preflight.items()
    ]
    command = [
        sys.executable,
        "-m",
        "big_vae.entrypoints.train",
        *base_production_overrides,
        *preflight_overrides,
    ]
    resolved_launch = _compose_resolved(command[3:])
    assert_exact_candidate_model(candidate, resolved_launch, context="production launch")
    _assert_resolved_contract(resolved_launch, pair_path)
    if production_scientific_config_sha256(resolved_launch) != scientific_config_sha256:
        raise RuntimeError("operational override changed the approved production scientific config")
    if resolved_launch.get("weightclip_launch_preflight") != launch_preflight:
        raise RuntimeError("Hydra production launch did not preserve the immutable preflight bindings")
    launch_contract_sha256 = canonical_fingerprint(resolved_launch)
    resolved_path = resolved_dir / (
        f"resolved_launch_{axes.name}_{launch_contract_sha256[:16]}.json"
    )
    resolved_sha = write_json_immutable(resolved_path, resolved_launch)
    print(
        "[weightclip-ae-train] approval_status="
        f"{approval['status']} approval_sha256={transition['approval_sha256']} "
        f"request_sha256={requirements.approval_request_sha256}",
        flush=True,
    )
    print(
        f"[weightclip-ae-train] data_mode=operator_bank pair_manifest={pair_path} "
        f"W_shape=128x128 X_shape=512x128 gauge_views=5 "
        f"canonical_probability={float(cfg['data_contract']['canonical_probability']):.6f}",
        flush=True,
    )
    print(f"[weightclip-ae-train] command={json.dumps(command)}", flush=True)
    print(f"[weightclip-ae-train] resolved_config={resolved_path} sha256={resolved_sha}", flush=True)
    print(
        "[weightclip-ae-train] disk_preflight="
        f"{disk_preflight['report_path']} sha256={disk_preflight['report_sha256']} "
        f"free={disk_preflight['observed_free_bytes']} "
        f"required={disk_preflight['required_free_bytes']} sufficient={disk_preflight['sufficient']}",
        flush=True,
    )
    if not args.execute:
        print("[weightclip-ae-train] stage=complete mode=verified-dry-run; pass --execute to launch", flush=True)
        return
    print("[weightclip-ae-train] stage=launch", flush=True)
    child_environment = _production_child_environment(approved_gpu_uuid)
    with _acquire_launch_storage(storage, disk_preflight), _acquire_gpu_launch_lease(
        approved_gpu_uuid,
        approved_peak_nvml_mib=approved_peak_nvml_mib,
    ) as gpu_lease:
        print(
            "[weightclip-ae-train] gpu_lease="
            f"uuid={approved_gpu_uuid} free_mib={gpu_lease['free_mib']:.1f} "
            f"required_mib={gpu_lease['required_free_mib']:.1f}",
            flush=True,
        )
        subprocess.run(command, check=True, env=child_environment)


if __name__ == "__main__":
    main()
