from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from big_vae.weightclip_benchmark.ae_scaling import canonical_fingerprint
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable


COMPARISON_SCHEMA = "weightclip_ae_runtime_comparison_v1"
LEGACY = "legacy_ratio_768_e24_d8_ffn4_l32"
DEEP = "deep_640_e32_d16_ffn4p6_l32"
DECODER = "decoder_heavy_864_e14_d20_ffn4_l32"
LATENT = "latent_800_e20_d10_ffn5p2_l64"
_RUN_PID = re.compile(r"_pid(?P<pid>[0-9]+)$")
HISTORICAL_COMMON = {
    "pair_manifest_sha256": "f3a26fc9772d43feabe138ef9fa5be5baf2f76323fb4f3079c12c5f65275f347",
    "source_implementation_seal_sha256": "4a8e4a37d9e836e6850e858cb051c34101dc2f15ae672e496fd893be6ec9976a",
    "candidate_artifact_set_sha256": "dda6400c1f5b6c895f137ab59e0722635f3ac9765d984c1d6b3d49d5e079d128",
    "candidate_report_sha256": "a3f0435e38fc35847cb8c690028186c7dd1a1324dd8be500dac5315e8fb72c6c",
    "candidate_index_sha256": "5d2aa4f7b5a73556608713535d1e049dddde39056803011a069018d0989bdec8",
}
HISTORICAL_MODEL_FINGERPRINTS = {
    LEGACY: "a2314118877304c10b11d32c8a707bd8081ac6d0cc459ad81ea5711c88cdc503",
    DEEP: "e9e913bf71daad3f36e7a6073933528a5d3ea891b2718b6814c995183de4d9d9",
    DECODER: "6f00b4848889d78ab167d92ab2c58ffe30d509440a735a6c9e3f9b7eca3b9fd7",
    LATENT: "fdee8e4a29dd35faf7a1c00c82469e78e93c8e004931957141c1d4fce0058144",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _copy_exact_immutable(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    data = source.read_bytes()
    source_sha = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError(f"immutable evidence conflict: {destination}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    destination.chmod(0o444)
    if sha256_file(destination) != source_sha:
        raise RuntimeError(f"evidence snapshot differs from source: {source}")
    return {
        "original_path": str(source),
        "original_sha256": source_sha,
        "snapshot_path": str(destination.resolve()),
        "snapshot_sha256": source_sha,
        "snapshot_bytes": len(data),
    }


def _success_evidence(
    path: Path,
    *,
    candidate: str,
    timings_path_override: Path | None = None,
    chain_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    overrides = {} if chain_overrides is None else dict(chain_overrides)
    contract_path = overrides.get(
        "profile_contract", Path(str(payload.get("profile_contract_path", "")))
    )
    resolved_path = overrides.get(
        "resolved_config", Path(str(payload.get("resolved_config_path", "")))
    )
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != payload.get("profile_contract_sha256")
        or not resolved_path.is_file()
        or sha256_file(resolved_path) != payload.get("resolved_config_sha256")
    ):
        raise RuntimeError(f"historical success contract/resolved config is missing/tampered: {path}")
    contract = _read_json(contract_path)
    source_path = overrides.get(
        "source_implementation_seal",
        Path(str(contract.get("source_implementation_seal_path", ""))),
    )
    if (
        not source_path.is_file()
        or sha256_file(source_path) != contract.get("source_implementation_seal_artifact_sha256")
    ):
        raise RuntimeError(f"historical success source seal is missing/tampered: {path}")
    timings_ref = payload.get("artifacts", {}).get("step_timings", {})
    jsonl_refs = [
        (Path(str(raw_path)), str(raw_sha))
        for raw_path, raw_sha in timings_ref.items()
        if str(raw_path).endswith(".jsonl")
    ]
    if len(jsonl_refs) != 1:
        raise RuntimeError(f"historical success summary lacks one step_timings.jsonl: {path}")
    declared_timings_path, timings_sha = jsonl_refs[0]
    timings_path = (
        declared_timings_path if timings_path_override is None else timings_path_override.resolve()
    )
    if not timings_path.is_file() or sha256_file(timings_path) != timings_sha:
        raise RuntimeError(f"historical step timings are missing/tampered: {timings_path}")
    rows = [json.loads(line) for line in timings_path.read_text(encoding="utf-8").splitlines()]
    steps = [int(row.get("global_step", -1)) for row in rows]
    measured = [row for row in rows if row.get("phase") == "measured"]
    produced = [int(row.get("produced_logical_index", -1)) for row in measured]
    finite_metrics = all(
        isinstance(row.get(key), (int, float))
        and float(row[key]) == float(row[key])
        and abs(float(row[key])) != float("inf")
        for row in measured
        for key in ("host_step_ms", "input_wait_ms", "loss")
    )
    bindings = {
        key: str(payload.get(key, ""))
        for key in (*HISTORICAL_COMMON, "model_fingerprint_sha256")
    }
    if (
        payload.get("candidate") != candidate
        or payload.get("termination_reason") != "bounded_profile_complete"
        or int(payload.get("executed_optimizer_steps", -1)) != 32
        or int(payload.get("warmup_steps", -1)) != 4
        or int(payload.get("measured_steps", -1)) != 28
        or bindings != {
            **HISTORICAL_COMMON,
            "model_fingerprint_sha256": HISTORICAL_MODEL_FINGERPRINTS[candidate],
        }
        or int(
            payload.get("production_input_path", {})
            .get("microbatch", {})
            .get("prepared_batch_queue_size", -1)
        )
        != 24
        or steps != list(range(1, 33))
        or [int(row["global_step"]) for row in measured] != list(range(5, 33))
        or not finite_metrics
        or any(current < previous for previous, current in zip(produced, produced[1:]))
        or int(payload.get("measured_producer_refill", {}).get("advance_tiles", -1)) <= 0
    ):
        raise RuntimeError(f"historical success summary has invalid contract: {path}")
    return {
        "candidate": candidate,
        "outcome": "historical_bounded_pass",
        "optimizer_steps_per_second": float(payload["optimizer_steps_per_second"]),
        "peak_nvml_used_mib": float(payload["peak_vram_mib"]["nvml_used"]),
        "pair_manifest_sha256": str(payload["pair_manifest_sha256"]),
        "source_implementation_seal_sha256": str(
            payload["source_implementation_seal_sha256"]
        ),
        "model_fingerprint_sha256": str(payload["model_fingerprint_sha256"]),
        "step_timings_sha256": timings_sha,
        "measured_rows": 28,
        "produced_cursor_monotonic_nondecreasing": True,
        "profile_contract_sha256": str(payload["profile_contract_sha256"]),
        "resolved_config_sha256": str(payload["resolved_config_sha256"]),
        "source_seal_artifact_sha256": str(
            contract["source_implementation_seal_artifact_sha256"]
        ),
    }


def _success_chain_paths(summary_path: Path) -> dict[str, Path]:
    summary = _read_json(summary_path)
    contract_path = Path(str(summary["profile_contract_path"]))
    contract = _read_json(contract_path)
    timings_path = next(
        Path(path)
        for path in summary["artifacts"]["step_timings"]
        if str(path).endswith(".jsonl")
    )
    return {
        "summary": summary_path,
        "step_timings": timings_path,
        "profile_contract": contract_path,
        "resolved_config": Path(str(summary["resolved_config_path"])),
        "source_implementation_seal": Path(
            str(contract["source_implementation_seal_path"])
        ),
    }


def _oom_evidence(
    config_path: Path,
    fatal_path: Path,
    launcher_failure_path: Path,
    *,
    candidate: str,
    enforce_live_inventory: bool = True,
) -> dict[str, Any]:
    config = _read_json(config_path)
    fatal = _read_json(fatal_path)
    preflight = config.get("weightclip_runtime_profile_preflight", {})
    bounded = config.get("train", {}).get("bounded_runtime_profile", {})
    run_match = _RUN_PID.search(Path(str(bounded.get("output_dir", ""))).name)
    traceback = str(fatal.get("traceback", ""))
    error = str(fatal.get("error", ""))
    launcher_failure = _read_json(launcher_failure_path)
    exact_stack = (
        "loss_for_backward.backward" in traceback
        if candidate == LATENT
        else "W_hat, mu, logvar, pred_dirs = model(" in traceback
    )
    profile_output = Path(str(bounded.get("output_dir", "")))
    if (
        preflight.get("candidate") != candidate
        or bounded.get("enabled") is not True
        or int(bounded.get("warmup_steps", -1)) != 4
        or int(bounded.get("measured_steps", -1)) != 28
        or fatal.get("role") != "train_launcher"
        or run_match is None
        or int(fatal.get("ppid", -1)) != int(run_match.group("pid"))
        or "CUDA out of memory" not in error
        or "OutOfMemoryError" not in traceback
        or not exact_stack
        or {
            "pair_manifest_sha256": str(
                config.get("train", {}).get("operator_bank", {}).get("pair_manifest_sha256", "")
            ),
            "source_implementation_seal_sha256": str(
                preflight.get("source_implementation_seal_sha256", "")
            ),
            "candidate_artifact_set_sha256": str(
                preflight.get("candidate_artifact_set_sha256", "")
            ),
            "candidate_report_sha256": str(preflight.get("candidate_report_sha256", "")),
            "candidate_index_sha256": str(preflight.get("candidate_index_sha256", "")),
            "model_fingerprint_sha256": str(preflight.get("model_fingerprint_sha256", "")),
        }
        != {
            **HISTORICAL_COMMON,
            "model_fingerprint_sha256": HISTORICAL_MODEL_FINGERPRINTS[candidate],
        }
        or launcher_failure
        != {
            "bounded_optimizer_steps": 32,
            "production_scheduler_steps": 500000,
            "return_code": 1,
            "status": "failed_worker_exit",
        }
        or (
            enforce_live_inventory
            and (
                (profile_output / "summary.json").exists()
                or any(profile_output.rglob("*.pt"))
                or any(profile_output.rglob("*.ckpt"))
                or any(profile_output.rglob("*.safetensors"))
            )
        )
    ):
        raise RuntimeError(f"historical OOM evidence does not bind candidate/run/error: {fatal_path}")
    phase = "first_backward" if "loss_for_backward.backward" in traceback else "first_forward"
    expected_phase = "first_backward" if candidate == LATENT else "first_forward"
    if phase != expected_phase:
        raise RuntimeError(
            f"historical OOM phase mismatch for {candidate}: expected={expected_phase} actual={phase}"
        )
    memory_match = re.search(r"has ([0-9.]+) GiB memory in use", error)
    if memory_match is None:
        raise RuntimeError("historical OOM evidence lacks memory-in-use measurement")
    return {
        "candidate": candidate,
        "outcome": "historical_artifact_established_cuda_oom",
        "failure_phase": phase,
        "memory_in_use_gib": float(memory_match.group(1)),
        "pair_manifest_sha256": str(config["train"]["operator_bank"]["pair_manifest_sha256"]),
        "source_implementation_seal_sha256": str(
            preflight["source_implementation_seal_sha256"]
        ),
        "profile_parent_pid": int(fatal["ppid"]),
        "model_fingerprint_sha256": str(preflight["model_fingerprint_sha256"]),
    }


def _profile_directory_inventory(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    root = Path(str(config["train"]["bounded_runtime_profile"]["output_dir"])).resolve()
    files = []
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    forbidden = [
        row["relative_path"]
        for row in files
        if row["relative_path"] == "summary.json"
        or Path(row["relative_path"]).suffix in {".pt", ".ckpt", ".safetensors"}
    ]
    if forbidden:
        raise RuntimeError(f"historical OOM profile directory has forbidden outputs: {forbidden}")
    return {
        "profile_output_dir": str(root),
        "files": files,
        "forbidden_summary_or_checkpoint_paths": [],
        "captured_before_comparison_materialization": True,
    }


def materialize_comparison(
    *,
    legacy_summary: Path,
    deep_summary: Path,
    decoder_config: Path,
    decoder_fatal: Path,
    decoder_launcher_failure: Path,
    latent_config: Path,
    latent_fatal: Path,
    latent_launcher_failure: Path,
    historical_candidate_index: Path,
    output_root: Path,
) -> Path:
    legacy = _success_evidence(legacy_summary, candidate=LEGACY)
    deep = _success_evidence(deep_summary, candidate=DEEP)
    sources = {
        LEGACY: _success_chain_paths(legacy_summary),
        DEEP: _success_chain_paths(deep_summary),
        DECODER: {
            "config_resolved": decoder_config,
            "fatal": decoder_fatal,
            "launcher_failure": decoder_launcher_failure,
        },
        LATENT: {
            "config_resolved": latent_config,
            "fatal": latent_fatal,
            "launcher_failure": latent_launcher_failure,
        },
    }
    outcomes = {
        LEGACY: legacy,
        DEEP: deep,
        DECODER: _oom_evidence(
            decoder_config, decoder_fatal, decoder_launcher_failure, candidate=DECODER
        ),
        LATENT: _oom_evidence(
            latent_config, latent_fatal, latent_launcher_failure, candidate=LATENT
        ),
    }
    candidate_index_payload = _read_json(historical_candidate_index)
    candidate_report_path = Path(str(candidate_index_payload.get("report", {}).get("path", "")))
    candidate_source_path = Path(
        str(candidate_index_payload.get("source_implementation", {}).get("path", ""))
    )
    if (
        sha256_file(historical_candidate_index) != HISTORICAL_COMMON["candidate_index_sha256"]
        or not candidate_report_path.is_file()
        or sha256_file(candidate_report_path) != HISTORICAL_COMMON["candidate_report_sha256"]
        or not candidate_source_path.is_file()
        or sha256_file(candidate_source_path)
        != candidate_index_payload.get("source_implementation", {}).get("sha256")
        or _read_json(candidate_source_path).get("source_implementation_seal_sha256")
        != HISTORICAL_COMMON["source_implementation_seal_sha256"]
        or candidate_index_payload.get("artifact_set_fingerprint_sha256")
        != HISTORICAL_COMMON["candidate_artifact_set_sha256"]
    ):
        raise RuntimeError("historical candidate index/report/source bundle is missing or mismatched")
    common_bundle_sources = {
        "candidate_index": historical_candidate_index,
        "candidate_report": candidate_report_path,
        "candidate_source_implementation": candidate_source_path,
    }
    common_bundle_contract = {
        name: {"original_path": str(path.resolve()), "original_sha256": sha256_file(path)}
        for name, path in common_bundle_sources.items()
    }
    oom_output_inventories = {
        DECODER: _profile_directory_inventory(decoder_config),
        LATENT: _profile_directory_inventory(latent_config),
    }
    evidence_source_contract = {
        candidate: {
            name: {"original_path": str(path.resolve()), "original_sha256": sha256_file(path)}
            for name, path in named.items()
        }
        for candidate, named in sources.items()
    }
    legacy_rate = float(outcomes[LEGACY]["optimizer_steps_per_second"])
    deep_rate = float(outcomes[DEEP]["optimizer_steps_per_second"])
    successful = [LEGACY, DEEP]
    winner = min(
        successful,
        key=lambda candidate: (-float(outcomes[candidate]["optimizer_steps_per_second"]), candidate),
    )
    runner = next(candidate for candidate in successful if candidate != winner)
    runner_to_winner_rate_ratio = (
        float(outcomes[runner]["optimizer_steps_per_second"])
        / float(outcomes[winner]["optimizer_steps_per_second"])
    )
    stress_candidates = sorted(
        candidate
        for candidate in successful
        if float(outcomes[candidate]["optimizer_steps_per_second"])
        >= 0.95 * float(outcomes[winner]["optimizer_steps_per_second"])
    )
    tool_path = Path(__file__).resolve()
    contract = {
        "schema": COMPARISON_SCHEMA,
        "comparison_tool_source": {"path": str(tool_path), "sha256": sha256_file(tool_path)},
        "evidence_source_contract": evidence_source_contract,
        "common_candidate_bundle_source_contract": common_bundle_contract,
        "outcomes": outcomes,
        "oom_profile_output_inventories": oom_output_inventories,
        "decision": {
            "status": "recommendation_only_pending_explicit_user_approval",
            "recommended_candidate": winner,
            "tie_band_relative": 0.05,
            "legacy_speed_advantage_over_deep_relative": legacy_rate / deep_rate - 1.0,
            "runner_candidate": runner,
            "runner_to_winner_rate_ratio": runner_to_winner_rate_ratio,
            "producer_stress_candidates": stress_candidates,
            "producer_stress_selection_rule": "rate_gte_0.95_times_fastest_rate",
            "reasons": [
                "legacy_is_faster_outside_the_5_percent_tie_band_frozen_before_current_confirmation",
                "decoder_and_latent_candidates_have_artifact_established_cuda_oom",
            ],
            "user_approval_recorded": False,
            "production_launch_authorized": False,
        },
        "limitations": {
            "historical_profile_source_is_not_current": True,
            "historical_runs_are_selection_evidence_not_current_approval_evidence": True,
            "current_selected_candidate_requires_new_production_exact_and_producer_stress_runs": True,
        },
    }
    fingerprint = canonical_fingerprint(contract)
    root = output_root.resolve() / f"comparison-{fingerprint[:16]}"
    snapshot_refs: dict[str, dict[str, Any]] = {}
    for candidate, named in sources.items():
        snapshot_refs[candidate] = {}
        for name, source in named.items():
            snapshot_refs[candidate][name] = _copy_exact_immutable(
                source, root / "evidence" / candidate / f"{name}.json"
            )
    tool_snapshot = _copy_exact_immutable(
        tool_path, root / "evidence" / "comparison_tool_source.py"
    )
    common_bundle_snapshots = {
        name: _copy_exact_immutable(
            source, root / "evidence" / "common_candidate_bundle" / f"{name}.json"
        )
        for name, source in common_bundle_sources.items()
    }
    report = {
        **contract,
        "comparison_fingerprint_sha256": fingerprint,
        "evidence_snapshots": snapshot_refs,
        "comparison_tool_snapshot": tool_snapshot,
        "common_candidate_bundle_snapshots": common_bundle_snapshots,
    }
    report_path = root / "comparison.json"
    write_json_immutable(report_path, report)
    validate_comparison_report(report_path)
    return report_path


def validate_comparison_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    fingerprint = str(report.get("comparison_fingerprint_sha256", ""))
    contract = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "comparison_fingerprint_sha256",
            "evidence_snapshots",
            "comparison_tool_snapshot",
            "common_candidate_bundle_snapshots",
        }
    }
    if (
        report.get("schema") != COMPARISON_SCHEMA
        or canonical_fingerprint(contract) != fingerprint
        or path.parent.name != f"comparison-{fingerprint[:16]}"
    ):
        raise RuntimeError("comparison artifact fingerprint/path contract mismatch")
    source_contract = report["evidence_source_contract"]
    if set(source_contract) != {LEGACY, DEEP, DECODER, LATENT}:
        raise RuntimeError("comparison evidence must contain exactly the four frozen candidates")
    tool_ref = report.get("comparison_tool_source", {})
    tool_snapshot = report.get("comparison_tool_snapshot", {})
    tool_snapshot_path = Path(str(tool_snapshot.get("snapshot_path", "")))
    if (
        not tool_snapshot_path.is_file()
        or sha256_file(tool_snapshot_path) != tool_ref.get("sha256")
        or tool_snapshot.get("snapshot_sha256") != tool_ref.get("sha256")
        or tool_snapshot_path.stat().st_mode & 0o222
    ):
        raise RuntimeError("comparison tool immutable source snapshot is missing/tampered")
    expected_snapshot_names = {
        LEGACY: {"summary", "step_timings", "profile_contract", "resolved_config", "source_implementation_seal"},
        DEEP: {"summary", "step_timings", "profile_contract", "resolved_config", "source_implementation_seal"},
        DECODER: {"config_resolved", "fatal", "launcher_failure"},
        LATENT: {"config_resolved", "fatal", "launcher_failure"},
    }
    if set(report.get("evidence_snapshots", {})) != set(expected_snapshot_names) or any(
        set(report["evidence_snapshots"].get(candidate, {})) != names
        for candidate, names in expected_snapshot_names.items()
    ):
        raise RuntimeError("comparison evidence snapshot key inventory is incomplete or has extras")
    common_expected = {"candidate_index", "candidate_report", "candidate_source_implementation"}
    common_snapshots = report.get("common_candidate_bundle_snapshots", {})
    common_contract = report.get("common_candidate_bundle_source_contract", {})
    if set(common_snapshots) != common_expected or set(common_contract) != common_expected:
        raise RuntimeError("comparison common candidate bundle snapshot inventory is invalid")
    for name in common_expected:
        snapshot = Path(str(common_snapshots[name]["snapshot_path"]))
        expected_sha = str(common_contract[name]["original_sha256"])
        if not snapshot.is_file() or sha256_file(snapshot) != expected_sha:
            raise RuntimeError("comparison common candidate bundle snapshot is missing/tampered")
    common_index = _read_json(
        Path(common_snapshots["candidate_index"]["snapshot_path"])
    )
    common_source_path = Path(
        common_snapshots["candidate_source_implementation"]["snapshot_path"]
    )
    common_report_path = Path(common_snapshots["candidate_report"]["snapshot_path"])
    if (
        sha256_file(common_source_path)
        != common_index.get("source_implementation", {}).get("sha256")
        or sha256_file(common_report_path) != common_index.get("report", {}).get("sha256")
        or _read_json(common_source_path).get("source_implementation_seal_sha256")
        != HISTORICAL_COMMON["source_implementation_seal_sha256"]
    ):
        raise RuntimeError("comparison common candidate bundle does not recompute from snapshots")
    for candidate, named in report.get("evidence_snapshots", {}).items():
        for name, ref in named.items():
            snapshot = Path(str(ref["snapshot_path"]))
            expected = str(ref["snapshot_sha256"])
            if not snapshot.is_file() or sha256_file(snapshot) != expected:
                raise RuntimeError(f"comparison evidence snapshot is missing/tampered: {snapshot}")
            if snapshot.stat().st_mode & 0o222:
                raise RuntimeError(f"comparison evidence snapshot is writable: {snapshot}")
            source_ref = source_contract[candidate][name]
            original = Path(str(source_ref["original_path"]))
            if original.exists() and sha256_file(original) != source_ref["original_sha256"]:
                raise RuntimeError(f"comparison original evidence drifted: {original}")
            if source_ref["original_sha256"] != expected:
                raise RuntimeError("comparison snapshot/original hashes disagree")
    if report["decision"].get("user_approval_recorded") is not False:
        raise RuntimeError("comparison artifact must not synthesize user approval")
    if (
        report["decision"].get("production_launch_authorized") is not False
        or report["decision"].get("status")
        != "recommendation_only_pending_explicit_user_approval"
        or float(report["decision"].get("tie_band_relative", -1.0)) != 0.05
    ):
        raise RuntimeError("comparison decision approval/tie-band contract is invalid")
    inventories = report.get("oom_profile_output_inventories", {})
    if set(inventories) != {DECODER, LATENT} or any(
        inventory.get("forbidden_summary_or_checkpoint_paths") != []
        for inventory in inventories.values()
    ):
        raise RuntimeError("comparison OOM output inventory is invalid")
    snapshots = report["evidence_snapshots"]
    recomputed = {
        LEGACY: _success_evidence(
            Path(snapshots[LEGACY]["summary"]["snapshot_path"]),
            candidate=LEGACY,
            timings_path_override=Path(snapshots[LEGACY]["step_timings"]["snapshot_path"]),
            chain_overrides={
                key: Path(snapshots[LEGACY][key]["snapshot_path"])
                for key in ("profile_contract", "resolved_config", "source_implementation_seal")
            },
        ),
        DEEP: _success_evidence(
            Path(snapshots[DEEP]["summary"]["snapshot_path"]),
            candidate=DEEP,
            timings_path_override=Path(snapshots[DEEP]["step_timings"]["snapshot_path"]),
            chain_overrides={
                key: Path(snapshots[DEEP][key]["snapshot_path"])
                for key in ("profile_contract", "resolved_config", "source_implementation_seal")
            },
        ),
        DECODER: _oom_evidence(
            Path(snapshots[DECODER]["config_resolved"]["snapshot_path"]),
            Path(snapshots[DECODER]["fatal"]["snapshot_path"]),
            Path(snapshots[DECODER]["launcher_failure"]["snapshot_path"]),
            candidate=DECODER,
            enforce_live_inventory=False,
        ),
        LATENT: _oom_evidence(
            Path(snapshots[LATENT]["config_resolved"]["snapshot_path"]),
            Path(snapshots[LATENT]["fatal"]["snapshot_path"]),
            Path(snapshots[LATENT]["launcher_failure"]["snapshot_path"]),
            candidate=LATENT,
            enforce_live_inventory=False,
        ),
    }
    if recomputed != report["outcomes"]:
        raise RuntimeError("comparison outcomes do not recompute from immutable snapshots")
    legacy_rate = float(recomputed[LEGACY]["optimizer_steps_per_second"])
    deep_rate = float(recomputed[DEEP]["optimizer_steps_per_second"])
    winner = min(
        (LEGACY, DEEP),
        key=lambda candidate: (-float(recomputed[candidate]["optimizer_steps_per_second"]), candidate),
    )
    runner = next(candidate for candidate in (LEGACY, DEEP) if candidate != winner)
    runner_ratio = (
        float(recomputed[runner]["optimizer_steps_per_second"])
        / float(recomputed[winner]["optimizer_steps_per_second"])
    )
    expected_stress_candidates = sorted(
        candidate
        for candidate in (LEGACY, DEEP)
        if float(recomputed[candidate]["optimizer_steps_per_second"])
        >= 0.95 * float(recomputed[winner]["optimizer_steps_per_second"])
    )
    if (
        report["decision"].get("recommended_candidate") != winner
        or abs(
            float(report["decision"].get("legacy_speed_advantage_over_deep_relative", -1.0))
            - (legacy_rate / deep_rate - 1.0)
        )
        > 1e-12
        or report["decision"].get("runner_candidate") != runner
        or abs(float(report["decision"].get("runner_to_winner_rate_ratio", -1.0)) - runner_ratio)
        > 1e-12
        or report["decision"].get("producer_stress_candidates")
        != expected_stress_candidates
        or report["decision"].get("producer_stress_selection_rule")
        != "rate_gte_0.95_times_fastest_rate"
    ):
        raise RuntimeError("comparison decision does not recompute from immutable outcomes")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize immutable historical AE profile evidence")
    parser.add_argument("--legacy-summary", type=Path, required=True)
    parser.add_argument("--deep-summary", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, required=True)
    parser.add_argument("--decoder-fatal", type=Path, required=True)
    parser.add_argument("--decoder-launcher-failure", type=Path, required=True)
    parser.add_argument("--latent-config", type=Path, required=True)
    parser.add_argument("--latent-fatal", type=Path, required=True)
    parser.add_argument("--latent-launcher-failure", type=Path, required=True)
    parser.add_argument("--historical-candidate-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = materialize_comparison(
        legacy_summary=args.legacy_summary,
        deep_summary=args.deep_summary,
        decoder_config=args.decoder_config,
        decoder_fatal=args.decoder_fatal,
        decoder_launcher_failure=args.decoder_launcher_failure,
        latent_config=args.latent_config,
        latent_fatal=args.latent_fatal,
        latent_launcher_failure=args.latent_launcher_failure,
        historical_candidate_index=args.historical_candidate_index,
        output_root=args.output_root,
    )
    print(f"[weightclip-ae-profile-comparison] report={report} sha256={sha256_file(report)}")


if __name__ == "__main__":
    main()
