from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping

import torch

from big_vae.weightclip_benchmark.ae_candidate_config import (
    ae_source_implementation_seal,
    assert_bounded_profile_production_parity,
    assert_exact_candidate_model,
    compose_resolved_train_config as _compose_resolved,
    production_scientific_config_sha256,
    resolve_ae_candidate,
)
from big_vae.weightclip_benchmark.ae_scaling import apply_scaling_axes, exact_parameter_ledger
from big_vae.weightclip_benchmark.contract import SOURCE_DATASETS
from big_vae.weightclip_benchmark.manifests import (
    sha256_file,
    validate_operator_bank_builder_source_seal,
    write_json_immutable,
)
from training.weightclip_benchmark.train_ae import (
    _acquire_gpu_launch_lease,
    _assert_resolved_contract,
    _candidate,
    _load_yaml,
    _operator_bank_overrides,
    _resolve_path,
    _validate_candidate_artifact_index,
    _validate_runtime_profile_summary,
)
from training.big_vae.worker import _resolve_profile_gpu_identity


PROFILE_SCHEMA = "weightclip_ae_runtime_v2"
PROFILE_MODES = ("production_exact", "producer_stress")
PROFILE_QUEUE_SIZE = {"production_exact": 24, "producer_stress": 4}
PRODUCTION_STEPS = 500_000
MAX_PROFILE_STEPS = 32
DEFAULT_WARMUP_STEPS = 4
DEFAULT_MEASURED_STEPS = 28
DEFAULT_TIMEOUT_SECONDS = 30 * 60
MAX_TIMEOUT_SECONDS = 60 * 60


def _assert_final_operator_bank(pair_path: Path) -> dict[str, Any]:
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    builder_source = payload.get("contract", {}).get("builder_source_implementation")
    if not isinstance(builder_source, dict):
        raise RuntimeError("operator-bank pair manifest lacks a builder source implementation seal")
    validate_operator_bank_builder_source_seal(builder_source)
    files = payload.get("contract", {}).get("checkpoint_files", [])
    if not isinstance(files, list):
        raise RuntimeError("operator-bank pair manifest has no checkpoint inventory")
    expected_datasets = {item.slug for item in SOURCE_DATASETS}
    actual_datasets = {str(row.get("dataset", "")) for row in files}
    if actual_datasets != expected_datasets:
        raise RuntimeError(
            f"runtime profile requires all ten source datasets: expected={sorted(expected_datasets)} "
            f"actual={sorted(actual_datasets)}"
        )
    if len(files) != 700:
        raise RuntimeError(f"runtime profile requires the final 700-checkpoint train bank, found {len(files)}")
    logical_identities = [
        (
            str(row.get("dataset", "")),
            str(row.get("lineage_id", "")),
            int(row.get("checkpoint_index_zero_based", -1)),
        )
        for row in files
    ]
    if len(set(logical_identities)) != 700:
        raise RuntimeError("runtime profile requires 700 unique dataset/lineage/checkpoint identities")
    canonical_checkpoint_paths = [str(Path(str(row.get("path", ""))).expanduser().resolve()) for row in files]
    if len(set(canonical_checkpoint_paths)) != 700:
        raise RuntimeError("runtime profile requires 700 distinct canonical checkpoint paths")
    for dataset in expected_datasets:
        rows = [row for row in files if row.get("dataset") == dataset]
        lineages = {str(row.get("lineage_id", "")) for row in rows}
        indices = {int(row.get("checkpoint_index_zero_based", -1)) for row in rows}
        if len(rows) != 70 or len(lineages) != 35 or indices != {43, 44}:
            raise RuntimeError(
                f"runtime profile bank inventory mismatch for {dataset}: "
                f"checkpoints={len(rows)} lineages={len(lineages)} indices={sorted(indices)}"
            )
        for lineage in lineages:
            lineage_indices = {
                int(row.get("checkpoint_index_zero_based", -1))
                for row in rows
                if str(row.get("lineage_id", "")) == lineage
            }
            if lineage_indices != {43, 44}:
                raise RuntimeError(
                    "runtime profile requires exact final checkpoint coverage per lineage: "
                    f"dataset={dataset} lineage={lineage} indices={sorted(lineage_indices)}"
                )
    protocol = payload.get("contract", {}).get("protocol", {})
    if list(protocol.get("checkpoint_splits", [])) != ["train"]:
        raise RuntimeError("runtime profile requires a train-only operator bank")
    for row in files:
        checkpoint_path = Path(str(row.get("path", ""))).expanduser()
        checkpoint_sha256 = str(row.get("sha256", ""))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha256:
            raise RuntimeError(
                "runtime profile checkpoint inventory path/SHA mismatch: "
                f"path={checkpoint_path} expected={checkpoint_sha256}"
            )
    for key in ("context_bank", "weight_tile_bank", "coverage_path"):
        path = Path(str(payload.get(key, "")))
        if not path.exists():
            raise RuntimeError(f"operator-bank artifact is missing: {key}={path}")
    coverage_path = Path(str(payload["coverage_path"]))
    coverage_sha256 = str(payload.get("coverage_sha256", ""))
    if sha256_file(coverage_path) != coverage_sha256:
        raise RuntimeError("operator-bank coverage artifact is not hash-bound")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not isinstance(coverage, dict) or not coverage:
        raise RuntimeError("operator-bank coverage artifact is empty or invalid")
    return payload


def assert_production_profile_parity(
    production: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    profile_mode: str = "production_exact",
) -> None:
    """Require exact scientific/runtime parity outside the explicit safety envelope."""

    if profile_mode not in PROFILE_MODES:
        raise RuntimeError(f"unknown bounded profile mode: {profile_mode!r}")
    production_queue = int(production["train"]["offline_batch_prefetch"]["queue_size"])
    profile_queue = int(profile["train"]["offline_batch_prefetch"]["queue_size"])
    if production_queue != 24 or profile_queue != PROFILE_QUEUE_SIZE[profile_mode]:
        raise RuntimeError(
            f"bounded profile queue mismatch: mode={profile_mode} production={production_queue} "
            f"actual={profile_queue} expected={PROFILE_QUEUE_SIZE[profile_mode]}"
        )
    normalized_profile = copy.deepcopy(dict(profile))
    normalized_profile["train"]["offline_batch_prefetch"]["queue_size"] = production_queue
    assert_bounded_profile_production_parity(production, normalized_profile)
    if int(profile["train"]["max_steps"]) != PRODUCTION_STEPS:
        raise RuntimeError("runtime profile changed the production scheduler horizon")


def _profile_overrides(
    *,
    output_dir: Path,
    warmup_steps: int,
    measured_steps: int,
    profile_mode: str = "production_exact",
    candidate: str,
    model_fingerprint_sha256: str,
    source_implementation_seal_sha256: str = "",
    candidate_artifact_set_sha256: str = "",
    candidate_report_sha256: str = "",
    candidate_index_sha256: str = "",
) -> list[str]:
    checkpoint_dir = output_dir / "forbidden_checkpoints"
    return [
        "+train.bounded_runtime_profile.enabled=true",
        f"+train.bounded_runtime_profile.schema={PROFILE_SCHEMA}",
        f"+train.bounded_runtime_profile.mode={profile_mode}",
        f"+train.bounded_runtime_profile.warmup_steps={warmup_steps}",
        f"+train.bounded_runtime_profile.measured_steps={measured_steps}",
        f"+train.bounded_runtime_profile.output_dir={output_dir}",
        f"train.offline_batch_prefetch.queue_size={PROFILE_QUEUE_SIZE[profile_mode]}",
        f"train.checkpoint_dir={checkpoint_dir}",
        "train.resume_state.enabled=false",
        "train.resume_state.auto_resume=false",
        "train.resume_checkpoint=",
        "train.telemetry.comet.enabled=false",
        "train.telemetry.wandb.enabled=false",
        f"+weightclip_runtime_profile_preflight.candidate={candidate}",
        f"+weightclip_runtime_profile_preflight.model_fingerprint_sha256={model_fingerprint_sha256}",
        "+weightclip_runtime_profile_preflight.source_implementation_seal_sha256="
        f"{source_implementation_seal_sha256}",
        "+weightclip_runtime_profile_preflight.candidate_artifact_set_sha256="
        f"{candidate_artifact_set_sha256}",
        "+weightclip_runtime_profile_preflight.candidate_report_sha256="
        f"{candidate_report_sha256}",
        "+weightclip_runtime_profile_preflight.candidate_index_sha256="
        f"{candidate_index_sha256}",
    ]


def _assert_gpu_idle(physical_uuid: str) -> None:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
        f"--id={physical_uuid}",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if rows:
        raise RuntimeError(f"cuda:0 has active compute processes; refusing a contaminated profile: {rows}")


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SECRET_TEXT = re.compile(
    r"("
    r"(?:^|[\s,{])[\"']?token[\"']?\s*[=:]\s*"
    r"|(?:bearer\s+)"
    r"|(?:authorization\s*[=:]\s*bearer\s+)"
    r"|(?:\b[A-Za-z0-9_-]*?(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token|bearer[_-]?token|hf[_-]?token|password|passwd|credentials?)\b\s*[=:]\s*)"
    r")[^\s,;]+",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ERROR_CLASS = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s")


def _sanitized_log_diagnostic(path: Path, *, max_lines: int = 80, max_chars: int = 12_000) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cleaned = _ANSI_ESCAPE.sub("", text)
    cleaned = _SECRET_TEXT.sub(r"\1<redacted>", cleaned)
    tail = "\n".join(cleaned.splitlines()[-max_lines:])[-max_chars:]
    classes = _ERROR_CLASS.findall(tail)
    return {
        "sanitized_tail": tail,
        "detected_error_class": classes[-1] if classes else None,
        "tail_max_lines": int(max_lines),
        "tail_max_chars": int(max_chars),
    }


def _redact_console_line(line: str) -> str:
    """Redact credential values before either console or artifact sees them."""

    return _SECRET_TEXT.sub(r"\1<redacted>", line)


def _run_worker_with_live_tee(
    command: list[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    log_path: Path,
) -> dict[str, Any]:
    """Run one worker with live console mirroring and an immutable complete log."""

    if log_path.exists():
        raise FileExistsError(f"worker console log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{log_path.name}.", suffix=".tmp", dir=log_path.parent
    )
    temporary = Path(temporary_name)
    timed_out = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", buffering=1) as log_handle:
            process = subprocess.Popen(
                command,
                env=dict(environment),
                text=True,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )

            def pump() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    redacted = _redact_console_line(line)
                    sys.stdout.write(redacted)
                    sys.stdout.flush()
                    log_handle.write(redacted)

            pump_thread = threading.Thread(target=pump, name="weightclip-profile-log-tee", daemon=True)
            pump_thread.start()
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                return_code = int(process.returncode if process.returncode is not None else -9)
            pump_thread.join(timeout=30.0)
            if pump_thread.is_alive():
                raise RuntimeError("bounded profile log tee did not drain after worker termination")
            log_handle.flush()
            os.fsync(log_handle.fileno())
        os.replace(temporary, log_path)
        log_path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "return_code": int(return_code),
        "timed_out": bool(timed_out),
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        **_sanitized_log_diagnostic(log_path),
    }
def _default_output_dir(
    config_path: Path,
    config: Mapping[str, Any],
    candidate: str,
    fingerprint: str,
    profile_mode: str,
) -> Path:
    profile_root = _resolve_path(str(config["profile_output_dir"]), relative_to=config_path.parent)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        profile_root
        / "runtime"
        / f"{candidate}_{fingerprint[:12]}_{profile_mode}_{stamp}_pid{os.getpid()}"
    ).resolve()


def _validate_completed_profile_summary(
    summary_path: Path,
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
    expected_optimizer_steps: int,
    expected_profile_mode: str = "production_exact",
    require_launcher_evidence: bool = True,
) -> dict[str, Any]:
    """Validate the worker result before the launcher may report completion."""
    validated = _validate_runtime_profile_summary(
        summary_path,
        candidate_name=candidate_name,
        model_fingerprint_sha256=model_fingerprint_sha256,
        trainable_parameters=trainable_parameters,
        pair_manifest_sha256=pair_manifest_sha256,
        production_scientific_sha256=production_scientific_sha256,
        source_implementation_seal_sha256=source_implementation_seal_sha256,
        candidate_artifact_set_sha256=candidate_artifact_set_sha256,
        candidate_report_sha256=candidate_report_sha256,
        candidate_index_sha256=candidate_index_sha256,
        production_resolved=production_resolved,
        expected_profile_mode=expected_profile_mode,
        require_launcher_evidence=require_launcher_evidence,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("executed_optimizer_steps", -1)) != expected_optimizer_steps:
        raise RuntimeError("bounded profile executed-step count disagrees with the hard cap")
    return validated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly bounded production-path 700M AE runtime profiler")
    parser.add_argument("--config", type=Path, default=Path("conf/weightclip_benchmark/ae_700m.yaml"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=PROFILE_MODES, required=True)
    parser.add_argument(
        "--candidate-artifact-index",
        type=Path,
        required=True,
        help="Content-addressed candidate artifact_index.json bound to the active source seal",
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        required=True,
        help="Exact finalized 700-checkpoint operator-bank pair manifest",
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Run the bounded profile; default is parity-checked dry run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_steps = int(args.warmup_steps) + int(args.measured_steps)
    if (
        int(args.warmup_steps) != DEFAULT_WARMUP_STEPS
        or int(args.measured_steps) != DEFAULT_MEASURED_STEPS
    ):
        raise RuntimeError(
            "the production-path bounded profile protocol is frozen at "
            f"warmup={DEFAULT_WARMUP_STEPS}, measured={DEFAULT_MEASURED_STEPS}"
        )
    if args.warmup_steps < 1 or args.measured_steps < 1 or total_steps > MAX_PROFILE_STEPS:
        raise RuntimeError(
            f"profile requires positive warmup/measured steps and at most {MAX_PROFILE_STEPS} total; got {total_steps}"
        )
    if not 1 <= int(args.timeout_seconds) <= MAX_TIMEOUT_SECONDS:
        raise RuntimeError(f"timeout_seconds must lie in [1, {MAX_TIMEOUT_SECONDS}]")

    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    if int(config.get("training_steps", -1)) != PRODUCTION_STEPS:
        raise RuntimeError("AE config no longer declares the frozen 500000-step production horizon")
    axes = _candidate(config, str(args.candidate))
    base_path = _resolve_path(str(config["base_model_config"]), relative_to=config_path.parent)
    candidate = resolve_ae_candidate(apply_scaling_axes(_load_yaml(base_path), axes))
    source_implementation = ae_source_implementation_seal()
    model_fingerprint = candidate.model_fingerprint_sha256
    parameter_ledger = exact_parameter_ledger(candidate.resolved_model_config)
    trainable_parameters = int(parameter_ledger["trainable_parameters"])
    candidate_artifacts = _validate_candidate_artifact_index(
        args.candidate_artifact_index.expanduser().resolve(),
        candidate_name=axes.name,
        model_fingerprint_sha256=model_fingerprint,
        trainable_parameters=trainable_parameters,
    )
    if (
        candidate_artifacts["source_implementation_seal_sha256"]
        != source_implementation["source_implementation_seal_sha256"]
    ):
        raise RuntimeError("candidate artifact index source seal differs from the runtime profiler source")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(config_path, config, axes.name, model_fingerprint, args.mode)
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"profile output directory is not empty: {output_dir}")

    bank_overrides, pair_path = _operator_bank_overrides(
        config,
        config_path,
        pair_manifest=args.pair_manifest,
    )
    _assert_final_operator_bank(pair_path)
    production_overrides = [
        *candidate.hydra_model_overrides,
        f"train.max_steps={PRODUCTION_STEPS}",
        "train.kl_beta=0.0",
        "train.kl_schedule.enabled=false",
        *bank_overrides,
    ]
    production_resolved = _compose_resolved(production_overrides)
    assert_exact_candidate_model(candidate, production_resolved, context="runtime-profile production reference")
    scientific_config_sha256 = production_scientific_config_sha256(production_resolved)
    _assert_resolved_contract(production_resolved, pair_path)
    safety_overrides = _profile_overrides(
        output_dir=output_dir,
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(args.measured_steps),
        profile_mode=args.mode,
        candidate=axes.name,
        model_fingerprint_sha256=model_fingerprint,
        source_implementation_seal_sha256=str(
            source_implementation["source_implementation_seal_sha256"]
        ),
        candidate_artifact_set_sha256=str(
            candidate_artifacts["artifact_set_fingerprint_sha256"]
        ),
        candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
        candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
    )
    profile_overrides = [*production_overrides, *safety_overrides]
    profile_resolved = _compose_resolved(profile_overrides)
    assert_exact_candidate_model(candidate, profile_resolved, context="bounded runtime profile")
    assert_production_profile_parity(
        production_resolved,
        profile_resolved,
        profile_mode=args.mode,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("bounded profile launcher must remain CUDA-uninitialized before GPU lease")
    gpu_identity = _resolve_profile_gpu_identity(torch.device("cuda:0"))
    if torch.cuda.is_initialized():
        raise RuntimeError("physical GPU identity resolution unexpectedly initialized CUDA")
    physical_gpu_uuid = str(gpu_identity["physical_uuid"])
    profile_contract = {
        "profile_schema": PROFILE_SCHEMA,
        "profile_mode": args.mode,
        "prefetch_queue_size": PROFILE_QUEUE_SIZE[args.mode],
        "candidate": axes.name,
        "model_fingerprint_sha256": model_fingerprint,
        "candidate_config_sha256": sha256_file(config_path),
        "base_model_config_sha256": sha256_file(base_path),
        "resolved_model_config_sha256": model_fingerprint,
        "production_scientific_config_sha256": scientific_config_sha256,
        "source_implementation_seal_sha256": source_implementation[
            "source_implementation_seal_sha256"
        ],
        "candidate_artifact_set_sha256": candidate_artifacts[
            "artifact_set_fingerprint_sha256"
        ],
        "candidate_report_path": candidate_artifacts["candidate_report_path"],
        "candidate_report_sha256": candidate_artifacts["candidate_report_sha256"],
        "candidate_index_path": candidate_artifacts["candidate_index_path"],
        "candidate_index_sha256": candidate_artifacts["candidate_index_sha256"],
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "production_max_steps": PRODUCTION_STEPS,
        "hard_cap_optimizer_steps": MAX_PROFILE_STEPS,
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": int(args.measured_steps),
        "external_tracking": False,
        "resume": False,
        "checkpoint_writes": False,
        "physical_gpu_uuid": physical_gpu_uuid,
        "cuda_visible_devices": physical_gpu_uuid,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    source_seal_path = output_dir / "source_implementation_seal.json"
    source_seal_artifact_sha256 = write_json_immutable(source_seal_path, source_implementation)
    profile_contract["source_implementation_seal_path"] = str(source_seal_path.resolve())
    profile_contract["source_implementation_seal_artifact_sha256"] = source_seal_artifact_sha256
    resolved_path = output_dir / "resolved_profile_config.json"
    resolved_sha = write_json_immutable(resolved_path, profile_resolved)
    profile_contract["resolved_profile_config"] = str(resolved_path.resolve())
    profile_contract["resolved_profile_config_sha256"] = resolved_sha
    contract_path = output_dir / "profile_contract.json"
    contract_sha = write_json_immutable(contract_path, profile_contract)
    command = [sys.executable, "-m", "big_vae.entrypoints.train", *profile_overrides]
    print(
        json.dumps(
            {
                "stage": "bounded_profile_preflight",
                "candidate": axes.name,
                "profile_mode": args.mode,
                "model_fingerprint_sha256": model_fingerprint,
                "pair_manifest": str(pair_path),
                "pair_manifest_sha256": sha256_file(pair_path),
                "production_scheduler_steps": PRODUCTION_STEPS,
                "bounded_optimizer_steps": total_steps,
                "timeout_seconds": int(args.timeout_seconds),
                "resolved_config": str(resolved_path),
                "resolved_config_sha256": resolved_sha,
                "profile_contract": str(contract_path),
                "profile_contract_sha256": contract_sha,
                "output_dir": str(output_dir),
                "command": command,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute:
        print("[weightclip-ae-runtime-profile] dry-run complete; pass --execute for the bounded run", flush=True)
        return

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_uuid
    environment["WEIGHTCLIP_AE_PROFILE_RESOLVED_SHA256"] = resolved_sha
    environment["WEIGHTCLIP_AE_PROFILE_CONTRACT_SHA256"] = contract_sha
    environment["WEIGHTCLIP_AE_PROFILE_GPU_UUID"] = physical_gpu_uuid
    with _acquire_gpu_launch_lease(
        physical_gpu_uuid,
        approved_peak_nvml_mib=0.0,
    ):
        run_result = _run_worker_with_live_tee(
            command,
            environment=environment,
            timeout_seconds=float(args.timeout_seconds),
            log_path=output_dir / "worker_console.log",
        )
    common_run_evidence = {
        "schema_version": 1,
        "profile_schema": PROFILE_SCHEMA,
        "profile_mode": args.mode,
        "candidate": axes.name,
        "model_fingerprint_sha256": model_fingerprint,
        "source_implementation_seal_sha256": source_implementation[
            "source_implementation_seal_sha256"
        ],
        "candidate_artifact_set_sha256": candidate_artifacts[
            "artifact_set_fingerprint_sha256"
        ],
        "candidate_index_sha256": candidate_artifacts["candidate_index_sha256"],
        "pair_manifest_sha256": sha256_file(pair_path),
        "profile_contract_sha256": contract_sha,
        "worker_console_log": {
            "path": run_result["log_path"],
            "sha256": run_result["log_sha256"],
        },
    }
    if run_result["timed_out"]:
        write_json_immutable(
            output_dir / "launcher_failure.json",
            {
                **common_run_evidence,
                "status": "failed_timeout",
                "timeout_seconds": int(args.timeout_seconds),
                "bounded_optimizer_steps": total_steps,
                "production_scheduler_steps": PRODUCTION_STEPS,
                "return_code": run_result["return_code"],
                "detected_error_class": run_result["detected_error_class"],
                "sanitized_log_tail": run_result["sanitized_tail"],
            },
        )
        raise RuntimeError(f"bounded AE runtime profile exceeded {args.timeout_seconds}s and was terminated")
    if int(run_result["return_code"]) != 0:
        write_json_immutable(
            output_dir / "launcher_failure.json",
            {
                **common_run_evidence,
                "status": "failed_worker_exit",
                "return_code": int(run_result["return_code"]),
                "bounded_optimizer_steps": total_steps,
                "production_scheduler_steps": PRODUCTION_STEPS,
                "detected_error_class": run_result["detected_error_class"],
                "sanitized_log_tail": run_result["sanitized_tail"],
            },
        )
        raise subprocess.CalledProcessError(int(run_result["return_code"]), command)

    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("bounded profile worker exited successfully without summary.json")
    try:
        validated_summary = _validate_completed_profile_summary(
            summary_path,
            candidate_name=axes.name,
            model_fingerprint_sha256=model_fingerprint,
            trainable_parameters=trainable_parameters,
            pair_manifest_sha256=sha256_file(pair_path),
            production_scientific_sha256=scientific_config_sha256,
            source_implementation_seal_sha256=str(
                source_implementation["source_implementation_seal_sha256"]
            ),
            candidate_artifact_set_sha256=str(
                candidate_artifacts["artifact_set_fingerprint_sha256"]
            ),
            candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
            candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
            production_resolved=production_resolved,
            expected_optimizer_steps=total_steps,
            expected_profile_mode=args.mode,
            require_launcher_evidence=False,
        )
    except Exception as exc:
        write_json_immutable(
            output_dir / "launcher_failure_post_validation.json",
            {
                "status": "failed_post_worker_validation",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
                "profile_contract_path": str(contract_path.resolve()),
                "profile_contract_sha256": contract_sha,
                "worker_console_log": common_run_evidence["worker_console_log"],
            },
        )
        raise
    launcher_result_path = output_dir / "launcher_result.json"
    write_json_immutable(
        launcher_result_path,
        {
            **common_run_evidence,
            "status": "validated_success",
            "return_code": 0,
            "summary": {
                "path": str(summary_path.resolve()),
                "sha256": sha256_file(summary_path),
            },
        },
    )
    # Re-run the public validator against the final, immutable launcher record.
    validated_summary = _validate_completed_profile_summary(
        summary_path,
        candidate_name=axes.name,
        model_fingerprint_sha256=model_fingerprint,
        trainable_parameters=trainable_parameters,
        pair_manifest_sha256=sha256_file(pair_path),
        production_scientific_sha256=scientific_config_sha256,
        source_implementation_seal_sha256=str(
            source_implementation["source_implementation_seal_sha256"]
        ),
        candidate_artifact_set_sha256=str(
            candidate_artifacts["artifact_set_fingerprint_sha256"]
        ),
        candidate_report_sha256=str(candidate_artifacts["candidate_report_sha256"]),
        candidate_index_sha256=str(candidate_artifacts["candidate_index_sha256"]),
        production_resolved=production_resolved,
        expected_optimizer_steps=total_steps,
        expected_profile_mode=args.mode,
        require_launcher_evidence=True,
    )
    print(
        "[weightclip-ae-runtime-profile] complete "
        f"mode={args.mode} summary={summary_path} "
        f"sha256={validated_summary['runtime_profile_summary_sha256']} "
        f"worker_log={run_result['log_path']} log_sha256={run_result['log_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
