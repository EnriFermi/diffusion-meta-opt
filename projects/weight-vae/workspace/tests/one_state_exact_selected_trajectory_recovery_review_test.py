from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

import scripts.review_one_state_exact_selected_trajectory_recovery as review


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return copy.deepcopy(review._load_frozen_manifest())


def _copy_source_staging(tmp_path: Path) -> Path:
    destination = tmp_path / "source.incomplete"
    destination.mkdir()
    for path in review.SOURCE_STAGING.iterdir():
        (destination / path.name).write_bytes(path.read_bytes())
    return destination


def _task_and_import_records() -> tuple[dict[str, object], dict[str, object]]:
    _, task = review._validate_task_manifest()
    _, imports = review._validate_import_manifest()
    task_record = {
        key: task[key]
        for key in (
            "pass",
            "manifest_sha256",
            "identity_gates",
            "source_dependency_count",
            "source_dependencies_sha256",
        )
    }
    import_record = {
        key: imports[key]
        for key in ("pass", "manifest_sha256", "module_count", "modules")
    }
    return task_record, import_record


def _startup_freeze() -> dict[str, object]:
    return {
        "live_source_sha256": review.EXPECTED_FINALIZER_SHA256,
        "live_normalized_source_sha256": (review.EXPECTED_FINALIZER_NORMALIZED_SHA256),
        "expected_normalized_source_sha256": (
            review.EXPECTED_FINALIZER_NORMALIZED_SHA256
        ),
        "recovery_protocol_sha256": review.EXPECTED_RECOVERY_PROTOCOL_SHA256,
    }


def _copied_freeze() -> dict[str, object]:
    return {
        **_startup_freeze(),
        "snapshot_source_sha256": review.EXPECTED_FINALIZER_SHA256,
        "snapshot_normalized_source_sha256": (
            review.EXPECTED_FINALIZER_NORMALIZED_SHA256
        ),
        "snapshot_protocol_sha256": review.EXPECTED_RECOVERY_PROTOCOL_SHA256,
    }


def _producer_reconstruction_input_files() -> dict[str, object]:
    from scripts import (
        finalize_one_state_exact_selected_trajectory_recovery as finalizer,
    )

    expected_hashes = {
        "config.json": finalizer.EXPECTED_ACCEPTED_RUN_CONFIG_SHA256,
        "weight_pool.pt": finalizer.EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256,
        "weight_pool_records.csv": finalizer.EXPECTED_ACCEPTED_RUN_RECORDS_SHA256,
        "vae_checkpoint.pt": finalizer.EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256,
    }
    assert finalizer.EXPECTED_ACCEPTED_RUN_DIR == review.ACCEPTED_RUN_DIR
    assert expected_hashes == review.EXPECTED_ACCEPTED_RUN_INPUT_SHA256
    run_audit = {
        "pass": True,
        "run_dir": str(finalizer.EXPECTED_ACCEPTED_RUN_DIR),
        "file_count": len(expected_hashes),
        "files": {
            name: {
                "path": str(finalizer.EXPECTED_ACCEPTED_RUN_DIR / name),
                "sha256": digest,
            }
            for name, digest in expected_hashes.items()
        },
    }
    return finalizer._require_reconstruction_input_pre_post_consistency(
        copy.deepcopy(run_audit),
        copy.deepcopy(run_audit),
    )


def _refreeze_finalizer(path: Path) -> tuple[str, str]:
    normalized = review._normalized_finalizer_sha256(path)
    prefix = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    replaced = 0
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f'{prefix}"{normalized}"\n'
            replaced += 1
    assert replaced == 1
    path.write_text("".join(lines), encoding="utf-8")
    assert review._normalized_finalizer_sha256(path) == normalized
    return _sha256(path), normalized


def _copy_refrozen_finalizer(path: Path) -> tuple[str, str]:
    source = (
        review.ROOT / "scripts/finalize_one_state_exact_selected_trajectory_recovery.py"
    )
    path.write_bytes(source.read_bytes())
    return _refreeze_finalizer(path)


def test_reviewer_source_never_imports_finalizer_or_producer() -> None:
    source = Path(review.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("finalize_one_state" in name for name in imports)
    assert not any(
        "run_one_state_exact_selected_trajectory" in name for name in imports
    )


def test_all_frozen_manifests_runtime_sources_and_finalizer_are_pinned() -> None:
    frozen = review._load_frozen_manifest()
    task, task_audit = review._validate_task_manifest()
    imports, import_audit = review._validate_import_manifest()
    finalizer = (
        review.ROOT / "scripts/finalize_one_state_exact_selected_trajectory_recovery.py"
    )
    assert _sha256(review.FROZEN_INPUT_MANIFEST) == (
        review.EXPECTED_FROZEN_INPUT_MANIFEST_SHA256
    )
    assert _sha256(review.RECOVERY_PROTOCOL) == (
        review.EXPECTED_RECOVERY_PROTOCOL_SHA256
    )
    assert _sha256(review.RECOVERY_TASK_MANIFEST) == (
        review.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
    )
    assert _sha256(review.RECOVERY_IMPORT_MANIFEST) == (
        review.EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
    )
    assert review.EXPECTED_FINALIZER_SHA256 == (
        "5407f712f46090eeeed004a76dea1ea10b250398171575586c1c6737b145b325"
    )
    assert review.EXPECTED_FINALIZER_NORMALIZED_SHA256 == (
        "e25be1074055a66ad68c63acc1f722191efbbf8a827ecd73adfeaa090329a599"
    )
    assert review.EXPECTED_ACCEPTED_RUN_INPUT_SHA256 == {
        "config.json": (
            "93d4b552f1bd9c682375d2b6967a430172bb5b45845156702e00d61399e82bb4"
        ),
        "weight_pool.pt": (
            "26c59c451ebe7439383521a1dec563dfa3de1f270b2f9063930b41a8202de7ef"
        ),
        "weight_pool_records.csv": (
            "98c120a9031fbcf564fb40b8a47ef6592eeb50fe947acf2f4304047e233ec933"
        ),
        "vae_checkpoint.pt": (
            "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"
        ),
    }
    assert _sha256(finalizer) == review.EXPECTED_FINALIZER_SHA256
    assert review._normalized_finalizer_sha256(finalizer) == (
        review.EXPECTED_FINALIZER_NORMALIZED_SHA256
    )
    assert task["expected_runtime"] == review.EXPECTED_RUNTIME
    assert task_audit["observed_runtime"] == review.EXPECTED_RUNTIME
    assert import_audit["module_count"] == 49
    assert imports["loaded_repository_modules"] == import_audit["modules"]
    assert frozen["recovery_dependencies"]["live_producer_source"]["sha256"] == (
        review.EXPECTED_PRODUCER_SHA256
    )


def test_exact_producer_reconstruction_schema_is_required_by_reviewer() -> None:
    record = _producer_reconstruction_input_files()
    result = review._validate_reconstruction_input_files(record)
    assert result == {
        "pass": True,
        "pre_post_equal": True,
        "run_dir": str(review.ACCEPTED_RUN_DIR),
        "file_count": 4,
        "files_sha256": review.EXPECTED_ACCEPTED_RUN_INPUT_SHA256,
    }
    assert "accepted_run_dir" in review.EXPECTED_CONFIG_KEYS
    assert "accepted_run_input_sha256" in review.EXPECTED_CONFIG_KEYS
    assert "reconstruction_input_files" in review.EXPECTED_RECOVERY_AUDIT_KEYS
    assert "reconstruction_input_files" in (
        review.EXPECTED_GEOMETRY_RECONSTRUCTION_KEYS
    )
    assert "reconstruction_inputs_exact_pre_post_load_run" in (
        review.EXPECTED_RECOVERY_VALIDITY_GATES
    )
    for omitted in record:
        incomplete = copy.deepcopy(record)
        incomplete.pop(omitted)
        with pytest.raises(review.ReviewError, match="audit schema"):
            review._validate_reconstruction_input_files(incomplete)


def test_top_level_geometry_and_validity_gate_reconstruction_linkage() -> None:
    record = _producer_reconstruction_input_files()
    recovery = {
        "reconstruction_input_files": record,
        "geometry_reconstruction": {
            "reconstruction_input_files": copy.deepcopy(record)
        },
        "recovery_validity_gates": {
            "reconstruction_inputs_exact_pre_post_load_run": True
        },
    }
    assert review._validate_reconstruction_input_linkage(recovery)["pass"] is True
    recovery["geometry_reconstruction"]["reconstruction_input_files"]["post_load_run"][
        "files"
    ]["config.json"]["sha256"] = "0" * 64
    with pytest.raises(review.ReviewError, match="top-level and geometry"):
        review._validate_reconstruction_input_linkage(recovery)


@pytest.mark.parametrize("filename", sorted(review.EXPECTED_ACCEPTED_RUN_INPUT_SHA256))
def test_each_reconstruction_input_hash_tamper_is_rejected(filename: str) -> None:
    record = _producer_reconstruction_input_files()
    for phase in ("pre_load_run", "post_load_run"):
        record[phase]["files"][filename]["sha256"] = "0" * 64
    with pytest.raises(review.ReviewError, match="identity mismatch"):
        review._validate_reconstruction_input_files(record)


def test_reconstruction_input_pre_post_mismatch_is_rejected() -> None:
    record = _producer_reconstruction_input_files()
    record["post_load_run"]["files"]["config.json"]["sha256"] = "0" * 64
    with pytest.raises(review.ReviewError, match="differ before and after"):
        review._validate_reconstruction_input_files(record)


def test_accepted_run_config_identity_rejects_path_and_hash_tamper() -> None:
    hashes = dict(review.EXPECTED_ACCEPTED_RUN_INPUT_SHA256)
    review._validate_accepted_run_identity(
        str(review.ACCEPTED_RUN_DIR), hashes, context="test config"
    )
    with pytest.raises(review.ReviewError, match="path mismatch"):
        review._validate_accepted_run_identity(
            str(review.ACCEPTED_RUN_DIR.with_name("wrong")),
            hashes,
            context="test config",
        )
    hashes["config.json"] = "0" * 64
    with pytest.raises(review.ReviewError, match="hashes mismatch"):
        review._validate_accepted_run_identity(
            str(review.ACCEPTED_RUN_DIR), hashes, context="test config"
        )


@pytest.mark.parametrize("mutation", ["extra", "hash", "symlink"])
def test_source_staging_is_exact_incomplete_and_rejects_tamper(
    tmp_path: Path, mutation: str
) -> None:
    source = _copy_source_staging(tmp_path)
    manifest = _manifest()
    manifest["source_staging_relative_path"] = str(source)
    if mutation == "extra":
        (source / "extra.txt").write_text("x\n", encoding="utf-8")
    elif mutation == "hash":
        (source / "state_metrics.csv").write_text("tampered\n", encoding="utf-8")
    else:
        target = source / "state_metrics.real"
        (source / "state_metrics.csv").rename(target)
        (source / "state_metrics.csv").symlink_to(target)
        manifest["expected_exact_file_set"].remove("state_metrics.csv")
        manifest["expected_exact_file_set"].append("state_metrics.real")
        manifest["files_sha256"]["state_metrics.real"] = manifest["files_sha256"].pop(
            "state_metrics.csv"
        )
    with pytest.raises(review.ReviewError):
        review._validate_source_staging(source, manifest)

    real_manifest = review._load_frozen_manifest()
    result = review._validate_source_staging(review.SOURCE_STAGING, real_manifest)
    assert len(result["files_sha256"]) == 14
    assert "INCOMPLETE" in result["files_sha256"]
    assert set(result["files_sha256"]).isdisjoint(
        review.FORBIDDEN_SOURCE_FINALIZATION_FILES
    )


def test_radius_rule_uses_only_5e_9_absolute_and_zero_relative_tolerance() -> None:
    proposals = pd.read_csv(review.SOURCE_STAGING / "proposal_diagnostics.csv")
    result = review._direction_diagnostics(proposals)
    assert result["directions_checked"] == 83
    assert result["maximum_direction_radius_absolute_error"] == pytest.approx(
        review.EXPECTED_MAX_RADIUS_ERROR, abs=5e-16
    )
    assert result["directions_above_original_tolerance"] == 37
    assert result["directions_above_recovery_tolerance"] == 0
    position = proposals.index[proposals["target_update"].astype(int).eq(74)][0]
    proposals.loc[position, "direction_norm"] = review.base_target_norm() + 6e-9
    with pytest.raises(review.ReviewError, match="maximum direction radius"):
        review._direction_diagnostics(proposals)


def _mutated_finalizer(tmp_path: Path, old: str, new: str, *, append: str = "") -> Path:
    source = (
        review.ROOT / "scripts/finalize_one_state_exact_selected_trajectory_recovery.py"
    ).read_text(encoding="utf-8")
    if old:
        assert old in source
        source = source.replace(old, new, 1)
    source += append
    path = tmp_path / "finalizer.py"
    path.write_text(source, encoding="utf-8")
    return path


def _static_review_mutation(path: Path, *, refreeze: bool = True) -> None:
    if refreeze:
        _refreeze_finalizer(path)
    review._validate_finalizer_static_source(
        path,
        expected_raw_sha256=_sha256(path),
        expected_normalized_sha256=review._normalized_finalizer_sha256(path),
    )


def test_finalizer_static_audit_accepts_exact_frozen_live_source() -> None:
    source = (
        review.ROOT / "scripts/finalize_one_state_exact_selected_trajectory_recovery.py"
    )
    result = review._validate_finalizer_static_source(source)
    assert result["raw_source_sha256"] == review.EXPECTED_FINALIZER_SHA256
    assert (
        result["normalized_source_sha256"]
        == review.EXPECTED_FINALIZER_NORMALIZED_SHA256
    )
    assert result["evaluate_call_sites"] == 1


@pytest.mark.parametrize(
    ("old", "new", "append", "message"),
    [
        (
            "",
            "",
            "\ndef _extra_evaluate(producer):\n    return producer._evaluate()\n",
            "exactly one",
        ),
        (
            "",
            "",
            "\ndef _forbidden(producer):\n"
            "    return producer._exact_joint_gradients()\n",
            "forbidden",
        ),
        (
            "_producer_dependency_preimport_audit(guard)",
            "None",
            "",
            "clean import gate",
        ),
        ('PRODUCTION_DEVICE = "cuda:0"', 'PRODUCTION_DEVICE = "cpu"', "", "device"),
    ],
)
def test_finalizer_static_audit_rejects_semantic_tamper(
    tmp_path: Path, old: str, new: str, append: str, message: str
) -> None:
    path = _mutated_finalizer(tmp_path, old, new, append=append)
    with pytest.raises(review.ReviewError, match=message):
        _static_review_mutation(path)


def test_finalizer_static_audit_rejects_false_normalized_self_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "finalizer.py"
    _, normalized = _copy_refrozen_finalizer(path)
    prefix = "EXPECTED_NORMALIZED_SOURCE_SHA256 = "
    source = path.read_text(encoding="utf-8")
    source = source.replace(f'{prefix}"{normalized}"', f'{prefix}"{"0" * 64}"', 1)
    path.write_text(source, encoding="utf-8")
    with pytest.raises(review.ReviewError, match="self-reported normalized"):
        review._validate_finalizer_static_source(
            path,
            expected_raw_sha256=_sha256(path),
            expected_normalized_sha256=normalized,
        )


def test_finalizer_snapshot_raw_hash_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "finalizer.py"
    expected_raw, expected_normalized = _copy_refrozen_finalizer(path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# raw snapshot tamper\n",
        encoding="utf-8",
    )
    with pytest.raises(review.ReviewError, match="raw hash"):
        review._validate_finalizer_static_source(
            path,
            expected_raw_sha256=expected_raw,
            expected_normalized_sha256=expected_normalized,
        )


def test_copied_source_and_all_four_recovery_snapshots_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    manifest = _manifest()
    for destination, source_name in review.COPIED_SOURCE_EVIDENCE.items():
        (output / destination).write_bytes(
            (review.SOURCE_STAGING / source_name).read_bytes()
        )
    finalizer_snapshot = output / "executed_recovery_finalizer_source_snapshot.py"
    finalizer_raw, finalizer_normalized = _copy_refrozen_finalizer(finalizer_snapshot)
    monkeypatch.setattr(review, "EXPECTED_FINALIZER_SHA256", finalizer_raw)
    monkeypatch.setattr(
        review, "EXPECTED_FINALIZER_NORMALIZED_SHA256", finalizer_normalized
    )
    fixed = {
        "frozen_continuation_reviewer_snapshot.py": (
            review.ROOT
            / "scripts/review_one_state_exact_selected_trajectory_continuation.py"
        ),
        "recovery_protocol_snapshot.md": review.RECOVERY_PROTOCOL,
        "recovery_frozen_input_manifest_snapshot.json": (review.FROZEN_INPUT_MANIFEST),
        "recovery_task_manifest_snapshot.json": review.RECOVERY_TASK_MANIFEST,
        "recovery_import_manifest_snapshot.json": review.RECOVERY_IMPORT_MANIFEST,
    }
    for destination, source in fixed.items():
        (output / destination).write_bytes(source.read_bytes())
    snapshot_hashes = {path.name: _sha256(path) for path in output.iterdir()}
    result = review._validate_copied_source_evidence(
        output,
        review.SOURCE_STAGING,
        manifest,
        {"snapshot_hashes": snapshot_hashes},
    )
    assert result == snapshot_hashes
    (output / "recovery_import_manifest_snapshot.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(review.ReviewError, match="snapshot mismatch"):
        review._validate_copied_source_evidence(
            output,
            review.SOURCE_STAGING,
            manifest,
            {"snapshot_hashes": snapshot_hashes},
        )


def _synthetic_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "recovery"
    output.mkdir(parents=True)
    for name in review.EXPECTED_MANIFESTED_ARTIFACTS:
        (output / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    finalizer_raw, finalizer_normalized = _copy_refrozen_finalizer(
        output / "executed_recovery_finalizer_source_snapshot.py"
    )
    monkeypatch.setattr(review, "EXPECTED_FINALIZER_SHA256", finalizer_raw)
    monkeypatch.setattr(
        review, "EXPECTED_FINALIZER_NORMALIZED_SHA256", finalizer_normalized
    )
    copies = {
        "recovery_protocol_snapshot.md": review.RECOVERY_PROTOCOL,
        "recovery_frozen_input_manifest_snapshot.json": (review.FROZEN_INPUT_MANIFEST),
        "recovery_task_manifest_snapshot.json": review.RECOVERY_TASK_MANIFEST,
        "recovery_import_manifest_snapshot.json": review.RECOVERY_IMPORT_MANIFEST,
    }
    for name, source in copies.items():
        (output / name).write_bytes(source.read_bytes())
    task_record, import_record = _task_and_import_records()
    config = {
        "protocol_id": review.RECOVERY_PROTOCOL_ID,
        "source_protocol_id": review.SOURCE_PROTOCOL_ID,
        "device": "cuda:0",
        "dtype": "float32 model/HVP; FP64 dense H/M replay products",
        "seed": "none; deterministic frozen terminal-state replay",
        "cache_mode": "accepted h2048 VAE plus frozen terminal progress",
        "source_staging": str(review.SOURCE_STAGING),
        "source_staging_mutation_allowed": False,
        "source_progress_checkpoint_sha256": (review.EXPECTED_SOURCE_PROGRESS_SHA256),
        "frozen_input_manifest_sha256": (review.EXPECTED_FROZEN_INPUT_MANIFEST_SHA256),
        "recovery_protocol_sha256": review.EXPECTED_RECOVERY_PROTOCOL_SHA256,
        "recovery_task_manifest_sha256": (
            review.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
        ),
        "recovery_import_manifest_sha256": (
            review.EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
        ),
        "producer_source_sha256": review.EXPECTED_PRODUCER_SHA256,
        "continuation_reviewer_source_sha256": (
            review.EXPECTED_CONTINUATION_REVIEWER_SHA256
        ),
        "geometry_evaluation_budget": 1,
        "optimization_gradient_evaluation_budget": 0,
        "proposal_budget": 0,
        "line_search_budget": 0,
        "parameter_update_budget": 0,
        "metric_replay_absolute_tolerance": review.REPLAY_ATOL,
        "spectrum_replay_absolute_tolerance": review.REPLAY_ATOL,
        "direction_radius_absolute_tolerance": review.RECOVERY_RADIUS_ATOL,
        "direction_radius_relative_tolerance": 0.0,
        "output_dir": str(output),
        "working_dir": str(output.with_name(f".{output.name}.incomplete.synthetic")),
        "finalizer_source_sha256": review.EXPECTED_FINALIZER_SHA256,
        "finalizer_normalized_source_sha256": (
            review.EXPECTED_FINALIZER_NORMALIZED_SHA256
        ),
        "startup_self_freeze": _startup_freeze(),
        "copied_self_freeze": _copied_freeze(),
        "task_manifest_preflight": task_record,
        "import_manifest_preflight": import_record,
        "copied_task_manifest_audit": task_record,
        "copied_import_manifest_audit": import_record,
        "expected_runtime": review.EXPECTED_RUNTIME,
        "accepted_run_dir": str(review.ACCEPTED_RUN_DIR),
        "accepted_run_input_sha256": dict(review.EXPECTED_ACCEPTED_RUN_INPUT_SHA256),
        "snapshot_hashes": {},
    }
    _json(output / "resolved_config.json", config)
    (output / "run.log").write_text("synthetic run\n", encoding="utf-8")
    artifacts = {
        name: _sha256(output / name) for name in review.EXPECTED_MANIFESTED_ARTIFACTS
    }
    _json(
        output / "artifact_manifest.json",
        {
            "protocol_id": review.RECOVERY_PROTOCOL_ID,
            "recovery_finalizer_source_sha256": review.EXPECTED_FINALIZER_SHA256,
            "recovery_finalizer_normalized_source_sha256": (
                review.EXPECTED_FINALIZER_NORMALIZED_SHA256
            ),
            "frozen_input_manifest_sha256": (
                review.EXPECTED_FROZEN_INPUT_MANIFEST_SHA256
            ),
            "recovery_task_manifest_sha256": (
                review.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
            ),
            "recovery_import_manifest_sha256": (
                review.EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
            ),
            "source_progress_checkpoint_sha256": (
                review.EXPECTED_SOURCE_PROGRESS_SHA256
            ),
            "artifacts": artifacts,
        },
    )
    _json(
        output / "FINALIZED.json",
        {
            "protocol_id": review.RECOVERY_PROTOCOL_ID,
            "source_protocol_id": review.SOURCE_PROTOCOL_ID,
            "status": "complete_awaiting_independent_recovery_review",
            "recovery_valid": True,
            "scientific_success": False,
            "accepted_updates": review.MAX_ACCEPTED_UPDATES,
            "new_accepted_updates": review.NEW_ACCEPTED_UPDATES,
            "termination": "max_updates_reached",
            "source_progress_checkpoint_sha256": (
                review.EXPECTED_SOURCE_PROGRESS_SHA256
            ),
            "accepted_vae_checkpoint_sha256": (
                review.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
            ),
            "recovery_task_manifest_sha256": (
                review.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
            ),
            "recovery_import_manifest_sha256": (
                review.EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256
            ),
            "recovery_finalizer_source_sha256": (review.EXPECTED_FINALIZER_SHA256),
            "recovery_finalizer_normalized_source_sha256": (
                review.EXPECTED_FINALIZER_NORMALIZED_SHA256
            ),
            "decision_sha256": _sha256(output / "decision.json"),
            "recovery_audit_sha256": _sha256(output / "recovery_audit.json"),
            "artifact_manifest_sha256": _sha256(output / "artifact_manifest.json"),
            "final_checkpoint_sha256": _sha256(output / "final_checkpoint.pt"),
        },
    )
    return output


def test_publication_exact_schema_hashes_and_regular_file_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _synthetic_publication(tmp_path, monkeypatch)
    result = review._validate_manifest_and_publication(output, _manifest())
    assert len(result["artifact_hashes"]) == 29
    assert len(review._regular_file_names(output)) == 32


@pytest.mark.parametrize("mutation", ["extra", "artifact", "finalized", "symlink"])
def test_publication_rejects_file_manifest_schema_and_symlink_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    output = _synthetic_publication(tmp_path, monkeypatch)
    if mutation == "extra":
        (output / "extra.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "artifact":
        (output / "decision.json").write_text("tampered\n", encoding="utf-8")
    elif mutation == "finalized":
        finalized = json.loads((output / "FINALIZED.json").read_text())
        finalized["extra"] = True
        _json(output / "FINALIZED.json", finalized)
    else:
        target = tmp_path / "outside.txt"
        target.write_text("outside\n", encoding="utf-8")
        (output / "decision.json").unlink()
        (output / "decision.json").symlink_to(target)
    with pytest.raises(review.ReviewError):
        review._validate_manifest_and_publication(output, _manifest())


def _model_snapshot(
    parameters: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor]
) -> dict[str, object]:
    digest = hashlib.sha256()
    result: dict[str, object] = {}
    total_tensors = 0
    total_elements = 0
    for category, tensors in (("parameters", parameters), ("buffers", buffers)):
        fingerprints: dict[str, object] = {}
        for name in sorted(tensors):
            fingerprint = review._tensor_fingerprint(tensors[name])
            fingerprints[name] = fingerprint
            digest.update(category.encode())
            digest.update(name.encode())
            digest.update(json.dumps(fingerprint, sort_keys=True).encode())
            total_tensors += 1
            total_elements += int(fingerprint["numel"])
        singular = category[:-1]
        result[category] = fingerprints
        result[f"{singular}_tensor_count"] = len(fingerprints)
        result[f"{singular}_element_count"] = sum(
            int(value["numel"]) for value in fingerprints.values()
        )
    result["all_tensor_count"] = total_tensors
    result["all_element_count"] = total_elements
    result["aggregate_sha256"] = digest.hexdigest()
    return result


def _task_record() -> dict[str, object]:
    gates = {
        "manifest_protocol": True,
        "source_weight_index": True,
        "record_task_name": True,
        "task_set_name": True,
        "tau": True,
        "full_train_sample_count": True,
        "full_test_sample_count": True,
        "selected_task_tensors": True,
        "accepted_checkpoint": True,
    }
    return {
        "pass": True,
        "gates": gates,
        "task_name": review.EXPECTED_TASK_NAME,
        "source_weight_index": review.SOURCE_WEIGHT_INDEX,
        "tau": review.EXPECTED_TASK_TAU,
        "train_sample_count": 16_384,
        "test_sample_count": 4_096,
        "selected_task_tensors": copy.deepcopy(review.EXPECTED_TASK_TENSORS),
        "accepted_vae_checkpoint_sha256": (
            review.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        ),
    }


def _reconstruction(
    *, active_hash: str, active_count: int
) -> tuple[dict[str, object], str]:
    base_parameters = {
        "layer.active": torch.zeros(2, 3, dtype=torch.float32),
        "layer.fixed": torch.ones(2, dtype=torch.float32),
    }
    installed_parameters = {
        "layer.active": torch.ones(2, 3, dtype=torch.float32),
        "layer.fixed": torch.ones(2, dtype=torch.float32),
    }
    buffers = {"running": torch.arange(2, dtype=torch.float32)}
    base = _model_snapshot(base_parameters, buffers)
    installed = _model_snapshot(installed_parameters, buffers)
    post = copy.deepcopy(installed)
    task = _task_record()
    runtime = {
        "pass": True,
        "expected_runtime": review.EXPECTED_RUNTIME,
        "observed_runtime": review.EXPECTED_RUNTIME,
        "requested_device": "cuda:0",
        "resolved_cuda_device_index": 0,
        "cuda_device_name": "synthetic",
        "cuda_compute_capability": [9, 0],
        "cuda_total_memory_bytes": 1,
        "cuda_current_device_index": 0,
        "cudnn_version": 1,
    }
    reconstruction = {
        "accepted_checkpoint_sha256": (review.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256),
        "expected_accepted_checkpoint_sha256": (
            review.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        ),
        "z_sha256": review.EXPECTED_Z_SHA256,
        "active_tensor_count": 1,
        "active_parameter_count": active_count,
        "installed_active_parameter_hash": active_hash,
        "post_replay_active_parameter_hash": active_hash,
        "base_parameter_gradient_slots": 0,
        "post_replay_parameter_gradient_slots": 0,
        "model_installation_audit": {
            "pass": True,
            "expected_changed_parameter_count": 1,
            "changed_parameter_count": 1,
            "changed_parameters": ["layer.active"],
            "changed_buffer_count": 0,
            "changed_buffers": [],
            "base_aggregate_sha256": base["aggregate_sha256"],
            "installed_aggregate_sha256": installed["aggregate_sha256"],
            "base_parameter_tensor_count": base["parameter_tensor_count"],
            "base_buffer_tensor_count": base["buffer_tensor_count"],
        },
        "model_replay_nonmutation_audit": {
            "pass": True,
            "all_parameters_bitwise_unchanged": True,
            "all_buffers_bitwise_unchanged": True,
            "parameter_tensor_count": installed["parameter_tensor_count"],
            "buffer_tensor_count": installed["buffer_tensor_count"],
            "installed_aggregate_sha256": installed["aggregate_sha256"],
            "post_replay_aggregate_sha256": installed["aggregate_sha256"],
        },
        "model_tensor_snapshots": {
            "base": base,
            "installed": installed,
            "post_replay": post,
        },
        "runtime_provenance": runtime,
        "reconstruction_input_files": _producer_reconstruction_input_files(),
        "task_at_load": task,
        "task_immediately_pre_replay": copy.deepcopy(task),
        "task_post_replay": copy.deepcopy(task),
        "full_ce_batch": True,
    }
    return reconstruction, str(installed["aggregate_sha256"])


def _checkpoint_gate_names() -> set[str]:
    return {
        "protocol_id",
        "source_protocol_id",
        "source_progress_checkpoint_sha256",
        "parent_checkpoint_sha256",
        "parent_progress_checkpoint_sha256",
        "selected_arm",
        "accepted_updates",
        "new_accepted_updates",
        "termination",
        "transition_chain",
        "active_hash",
        "stored_metrics",
        "source_weight_index",
        "z_sha256",
        "accepted_vae_checkpoint_sha256",
    }


def _geometry_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object]]:
    output = tmp_path / "recovery"
    output.mkdir(parents=True)
    active_state = {"layer.active": torch.ones(2, 3, dtype=torch.float32)}
    active_hash, active_count = review._named_tensor_hash(active_state)
    monkeypatch.setattr(review, "EXPECTED_ACTIVE_PARAMETER_HASH", active_hash)
    monkeypatch.setattr(review, "ACTIVE_PARAMETER_COUNT", active_count)
    monkeypatch.setattr(review, "ACTIVE_TENSOR_COUNT", 1)

    target = torch.linspace(0.01, 2.0, review.DIMENSION, dtype=torch.float64)
    hessian = torch.diag(target.sqrt())
    matrix = hessian @ hessian.T
    eig = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
    low_mask = eig < review.GEOMETRY_LOW_THRESHOLD
    low_basis = torch.eye(review.DIMENSION, dtype=torch.float64)[:, low_mask]
    low_projector = low_basis @ low_basis.T
    closures = review._recompute_geometry_metrics(
        hessian=hessian,
        matrix=matrix,
        eig=eig,
        low_basis=low_basis,
        low_projector=low_projector,
    )
    projector_hash = review._sha256_tensor(low_projector)
    metrics: dict[str, object] = {
        **closures,
        "task_loss": 0.75,
        "hessian_sec": 0.25,
        "low_basis_hash": projector_hash,
    }
    final_row = {
        "accepted_update": review.MAX_ACCEPTED_UPDATES,
        "parameter_hash": active_hash,
        "parent_frozen_low_energy": 0.1,
        "phase": "relaxed_continuation",
        **metrics,
    }
    progress = {
        "active_model_state": active_state,
        "state_rows": [final_row],
    }
    spectra = pd.DataFrame(
        {
            "accepted_update": review.MAX_ACCEPTED_UPDATES,
            "rank": np.arange(review.DIMENSION),
            "m_eigenvalue": eig.numpy(),
            "a_contribution": np.square(eig.numpy() - 1.0),
            "phase": "relaxed_continuation",
        }
    )
    reconstruction, model_aggregate = _reconstruction(
        active_hash=active_hash, active_count=active_count
    )
    tensor_values = {
        "hessian": hessian,
        "matrix": matrix,
        "eig": eig,
        "current_low_basis": low_basis,
        "current_low_projector": low_projector,
    }
    fingerprints = {
        name: review._tensor_fingerprint(value) for name, value in tensor_values.items()
    }
    geometry_payload = {
        "protocol_id": review.RECOVERY_PROTOCOL_ID,
        "source_protocol_id": review.SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": (review.EXPECTED_SOURCE_PROGRESS_SHA256),
        "accepted_vae_checkpoint_sha256": (
            review.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        ),
        "active_parameter_hash": active_hash,
        "source_weight_index": review.SOURCE_WEIGHT_INDEX,
        "task_name": review.EXPECTED_TASK_NAME,
        "tau": review.EXPECTED_TASK_TAU,
        "z_sha256": review.EXPECTED_Z_SHA256,
        "accepted_update": review.MAX_ACCEPTED_UPDATES,
        "geometry_constants": {
            "dimension": review.DIMENSION,
            "epsilon": review.GEOMETRY_EPSILON,
            "old_beta": review.GEOMETRY_OLD_BETA,
            "low_threshold": review.GEOMETRY_LOW_THRESHOLD,
            "absolute_tolerance": review.REPLAY_ATOL,
        },
        **tensor_values,
        "metrics": metrics,
        "tensor_fingerprints": fingerprints,
        "installed_model_aggregate_sha256": model_aggregate,
        "post_replay_model_aggregate_sha256": model_aggregate,
    }
    torch.save(geometry_payload, output / "replayed_final_geometry.pt")
    with (output / "replayed_final_spectrum.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rank", "stored", "replayed", "abs_error"]
        )
        writer.writeheader()
        for rank, value in enumerate(eig.tolist()):
            writer.writerow(
                {
                    "rank": rank,
                    "stored": value,
                    "replayed": value,
                    "abs_error": 0.0,
                }
            )
    numeric_names = sorted(set(metrics) - {"hessian_sec", "low_basis_hash"})
    metric_errors = {name: 0.0 for name in numeric_names}
    metadata_gates = {
        name: True
        for name in (
            "protocol_id",
            "source_protocol_id",
            "source_progress",
            "accepted_checkpoint",
            "active_parameter_hash",
            "source_weight_index",
            "task_name",
            "tau",
            "z_sha256",
            "accepted_update",
            "model_nonmutation",
            "constants",
        )
    }
    artifact_audit = {
        "pass": True,
        "metadata_gates": metadata_gates,
        "matrix_from_hessian_max_abs_error": 0.0,
        "matrix_symmetry_max_abs_error": 0.0,
        "eigvalsh_matrix_max_abs_error": 0.0,
        "stored_spectrum_max_abs_error": 0.0,
        "low_projector_basis_max_abs_error": 0.0,
        "low_projector_sha256": projector_hash,
        "low_basis_orthogonality_max_abs": closures["low_basis_orthogonality_max_abs"],
        "low_basis_eigen_residual_relative": closures[
            "low_basis_eigen_residual_relative"
        ],
        "metric_closure_errors": {name: 0.0 for name in closures},
        "metric_closure_max_abs_error": 0.0,
        "source_metric_max_abs_error": 0.0,
        "tensor_fingerprints": fingerprints,
        "artifact_sha256": _sha256(output / "replayed_final_geometry.pt"),
        "cpu_reload_pass": True,
    }
    replay = {
        "pass": True,
        "metric_absolute_tolerance": review.REPLAY_ATOL,
        "metric_errors": metric_errors,
        "metric_max_absolute_error": 0.0,
        "metric_max_error_name": numeric_names[0],
        "spectrum_absolute_tolerance": review.REPLAY_ATOL,
        "spectrum_eigenvalue_count": review.DIMENSION,
        "spectrum_max_absolute_error": 0.0,
        "spectrum_max_error_rank": 0,
        "low_basis_hash_matches": True,
        "hessian_shape": [review.DIMENSION, review.DIMENSION],
        "matrix_shape": [review.DIMENSION, review.DIMENSION],
        "spectrum_shape": [review.DIMENSION],
        "low_basis_shape": list(low_basis.shape),
        "low_projector_shape": [review.DIMENSION, review.DIMENSION],
        "hessian_sha256": fingerprints["hessian"]["sha256"],
        "matrix_sha256": fingerprints["matrix"]["sha256"],
        "spectrum_tensor_sha256": fingerprints["eig"]["sha256"],
        "low_basis_sha256": fingerprints["current_low_basis"]["sha256"],
        "low_projector_sha256": projector_hash,
        "active_parameter_hash_matches": True,
        "replayed_final_spectrum_sha256": _sha256(
            output / "replayed_final_spectrum.csv"
        ),
        "replayed_final_spectrum_rows": review.DIMENSION,
        "geometry_artifact_sha256": _sha256(output / "replayed_final_geometry.pt"),
        "geometry_artifact_audit": artifact_audit,
    }
    _json(output / "decision.json", {"final_metrics": metrics})
    final = {
        "protocol_id": review.RECOVERY_PROTOCOL_ID,
        "source_protocol_id": review.SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": (review.EXPECTED_SOURCE_PROGRESS_SHA256),
        "parent_checkpoint_sha256": review.EXPECTED_PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": (review.EXPECTED_PARENT_PROGRESS_SHA256),
        "selected_arm": "low",
        "accepted_updates": review.MAX_ACCEPTED_UPDATES,
        "new_accepted_updates": review.NEW_ACCEPTED_UPDATES,
        "termination": "max_updates_reached",
        "transition_chain_sha256": review.EXPECTED_TRANSITION_CHAIN_SHA256,
        "active_parameter_hash": active_hash,
        "active_model_state": active_state,
        "source_weight_index": review.SOURCE_WEIGHT_INDEX,
        "z_sha256": review.EXPECTED_Z_SHA256,
        "accepted_vae_checkpoint_sha256": (
            review.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        ),
        "stored_state_metrics": final_row,
    }
    torch.save(final, output / "final_checkpoint.pt")
    checkpoint_hash = _sha256(output / "final_checkpoint.pt")
    audit = {
        "geometry_execution": {
            "geometry_evaluation_count": 1,
            "forbidden_operation_attempts": 0,
            "optimization_gradient_evaluations": 0,
            "proposals": 0,
            "line_searches": 0,
            "parameter_updates": 0,
        },
        "geometry_reconstruction": reconstruction,
        "reconstruction_input_files": copy.deepcopy(
            reconstruction["reconstruction_input_files"]
        ),
        "final_geometry_replay": replay,
        "checkpoint_lineage": {
            "pass": True,
            "bitwise_tensor_equality": True,
            "active_tensor_count": 1,
            "active_parameter_count": active_count,
            "active_parameter_hash": active_hash,
            "metadata_gates": {name: True for name in _checkpoint_gate_names()},
            "final_checkpoint_sha256": checkpoint_hash,
            "cpu_reload_pass": True,
        },
    }
    return output, {"progress": progress, "spectra": spectra}, audit


def test_independent_geometry_checkpoint_and_nonmutation_packet_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, packet, audit = _geometry_packet(tmp_path, monkeypatch)
    result = review._validate_replay_and_final_checkpoint(output, packet, audit)
    assert result["metric_max_abs_error"] == 0.0
    assert result["spectrum_max_abs_error"] == 0.0
    assert result["geometry"]["spectrum"]["row_count"] == review.DIMENSION
    assert result["reconstruction"]["buffer_count"] == 1


@pytest.mark.parametrize(
    ("tensor_name", "index", "message"),
    [
        ("hessian", (0, 0), "stored M"),
        ("matrix", (0, 0), "stored M"),
        ("eig", (0,), "eigenvalue tensor"),
        ("current_low_basis", (0, 0), "projector"),
    ],
)
def test_geometry_rejects_h_m_eig_and_basis_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_name: str,
    index: tuple[int, ...],
    message: str,
) -> None:
    output, packet, audit = _geometry_packet(tmp_path, monkeypatch)
    payload = torch.load(
        output / "replayed_final_geometry.pt", map_location="cpu", weights_only=True
    )
    payload[tensor_name][index] += 1e-4
    payload["tensor_fingerprints"][tensor_name] = review._tensor_fingerprint(
        payload[tensor_name]
    )
    torch.save(payload, output / "replayed_final_geometry.pt")
    with pytest.raises(review.ReviewError, match=message):
        review._validate_replayed_geometry(
            output, packet, audit["final_geometry_replay"]
        )


def test_all_512_spectrum_rows_are_checked_not_only_reported_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, packet, audit = _geometry_packet(tmp_path, monkeypatch)
    rows = list(
        csv.DictReader(
            (output / "replayed_final_spectrum.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    rows[311]["replayed"] = str(float(rows[311]["replayed"]) + 1e-4)
    with (output / "replayed_final_spectrum.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rank", "stored", "replayed", "abs_error"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(review.ReviewError, match="rank 311"):
        review._validate_replayed_geometry(
            output, packet, audit["final_geometry_replay"]
        )


def test_final_checkpoint_bitwise_lineage_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, packet, audit = _geometry_packet(tmp_path, monkeypatch)
    final = torch.load(
        output / "final_checkpoint.pt", map_location="cpu", weights_only=True
    )
    final["active_model_state"]["layer.active"][0, 0] += 1.0
    torch.save(final, output / "final_checkpoint.pt")
    with pytest.raises(review.ReviewError, match="tensor lineage"):
        review._validate_replay_and_final_checkpoint(output, packet, audit)


@pytest.mark.parametrize(
    ("counter", "value"),
    [
        ("geometry_evaluation_count", 2),
        ("optimization_gradient_evaluations", 1),
        ("parameter_updates", 1),
    ],
)
def test_runtime_counter_tamper_is_rejected(counter: str, value: int) -> None:
    execution = {
        "geometry_evaluation_count": 1,
        "forbidden_operation_attempts": 0,
        "optimization_gradient_evaluations": 0,
        "proposals": 0,
        "line_searches": 0,
        "parameter_updates": 0,
    }
    review._validate_geometry_execution(execution)
    execution[counter] = value
    with pytest.raises(review.ReviewError, match="exact-one-evaluate"):
        review._validate_geometry_execution(execution)


def test_task_audit_tamper_is_rejected() -> None:
    record = _task_record()
    review._validate_task_reconstruction_record(record)
    record["selected_task_tensors"]["test_labels"]["sha256"] = "0" * 64
    with pytest.raises(review.ReviewError, match="task audit identity"):
        review._validate_task_reconstruction_record(record)


def test_import_closure_tamper_is_rejected() -> None:
    manifest, audit = review._validate_import_manifest()
    record = {
        "pass": True,
        "module_count": 49,
        "modules": manifest["loaded_repository_modules"],
        "allowed_additions": [
            "__main__",
            "scripts.finalize_one_state_exact_selected_trajectory_recovery",
            "__mp_main__",
        ],
    }
    review._validate_runtime_import_closure(record, manifest)
    tampered = copy.deepcopy(record)
    first = next(iter(tampered["modules"]))
    tampered["modules"][first]["sha256"] = "0" * 64
    with pytest.raises(review.ReviewError, match="import closure"):
        review._validate_runtime_import_closure(tampered, manifest)


def test_runtime_import_closure_matches_verified_finalizer_mp_main_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import (
        finalize_one_state_exact_selected_trajectory_recovery as finalizer,
    )

    manifest, _ = review._validate_import_manifest()
    loaded_modules = manifest["loaded_repository_modules"]
    synthetic_modules: dict[str, types.ModuleType] = {}
    for module_name, entry in loaded_modules.items():
        module = types.ModuleType(module_name)
        module.__file__ = str(review.ROOT / entry["relative_path"])
        synthetic_modules[module_name] = module
    finalizer_module_name = (
        "scripts.finalize_one_state_exact_selected_trajectory_recovery"
    )
    synthetic_modules.update(
        {
            "__main__": finalizer,
            finalizer_module_name: finalizer,
            "__mp_main__": finalizer,
        }
    )
    monkeypatch.setattr(
        finalizer, "sys", types.SimpleNamespace(modules=synthetic_modules)
    )

    record = finalizer._audit_loaded_repository_module_closure(
        repository_root=review.ROOT,
        import_manifest=manifest,
    )
    assert record["allowed_additions"] == [
        "__main__",
        finalizer_module_name,
        "__mp_main__",
    ]
    assert review._validate_runtime_import_closure(record, manifest) == record

    legacy_record = copy.deepcopy(record)
    legacy_record["allowed_additions"].remove("__mp_main__")
    with pytest.raises(review.ReviewError, match="import closure"):
        review._validate_runtime_import_closure(legacy_record, manifest)


def test_parameter_buffer_nonmutation_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, audit = _geometry_packet(tmp_path, monkeypatch)
    geometry = torch.load(
        output / "replayed_final_geometry.pt", map_location="cpu", weights_only=True
    )
    reconstruction = audit["geometry_reconstruction"]
    review._validate_geometry_reconstruction(reconstruction, geometry)
    tampered = copy.deepcopy(reconstruction)
    changed_post = _model_snapshot(
        {
            "layer.active": torch.ones(2, 3),
            "layer.fixed": torch.ones(2),
        },
        {"running": torch.tensor([0.0, 9.0])},
    )
    tampered["model_tensor_snapshots"]["post_replay"] = changed_post
    with pytest.raises(review.ReviewError, match="fingerprint changed"):
        review._validate_geometry_reconstruction(tampered, geometry)


def _write_nonblank_png(path: Path, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    for index in range(20):
        draw.line(
            (0, index * height // 24, width - 1, height - 1 - index * 9),
            fill=(index * 10, 20, 255 - index * 10),
            width=3,
        )
    image.save(path)


def _synthetic_plot_packet() -> dict[str, object]:
    states = pd.DataFrame(
        {
            "accepted_update": np.arange(review.MAX_ACCEPTED_UPDATES + 1),
            "exact_a_per_dim": np.linspace(1.0, 0.9, 101),
            "damped_full_burg_per_dim": np.linspace(6.0, 5.0, 101),
            "a_low90_abs_per_dim": np.linspace(0.8, 0.7, 101),
            "a_gt1": np.linspace(0.2, 0.1, 101),
            "m_p50": np.linspace(0.01, 0.02, 101),
            "effective_rank": np.linspace(10.0, 20.0, 101),
            "phase": ["i6"] * 18 + ["relaxed_continuation"] * 83,
        }
    )
    spectra_rows = []
    for update in range(101):
        for rank in range(review.DIMENSION):
            value = rank / review.DIMENSION
            spectra_rows.append(
                {
                    "accepted_update": update,
                    "rank": rank,
                    "m_eigenvalue": value,
                    "a_contribution": (value - 1.0) ** 2,
                    "phase": (
                        "i6"
                        if update <= review.START_UPDATE
                        else "relaxed_continuation"
                    ),
                }
            )
    proposals = pd.DataFrame(
        {
            "target_update": np.arange(1, 101),
            "direction_norm": np.full(100, review.base_target_norm()),
        }
    )
    return {
        "states": states,
        "spectra": pd.DataFrame(spectra_rows),
        "proposals": proposals,
    }


def test_png_dimensions_and_plot_backing_data_are_checked(tmp_path: Path) -> None:
    packet = _synthetic_plot_packet()
    output = tmp_path / "plots"
    output.mkdir()
    finalizer = (
        review.ROOT / "scripts/finalize_one_state_exact_selected_trajectory_recovery.py"
    )
    (output / "executed_recovery_finalizer_source_snapshot.py").write_bytes(
        finalizer.read_bytes()
    )
    dimensions = {
        "recovery_trajectory.png": (3060, 1800),
        "recovery_spectra.png": (1800, 1080),
        "recovery_direction_radius_audit.png": (1980, 1080),
    }
    for name, size in dimensions.items():
        _write_nonblank_png(output / name, size)
    result = review._validate_plot_data_and_pngs(output, packet)
    assert result["trajectory_rows"] == 101
    assert result["spectra_rows"] == 3 * review.DIMENSION
    assert result["direction_rows"] == review.NEW_ACCEPTED_UPDATES
    packet["states"].loc[50, "exact_a_per_dim"] = np.nan
    with pytest.raises(review.ReviewError, match="trajectory plot backing"):
        review._validate_plot_data_and_pngs(output, packet)


def test_review_json_is_one_new_file_outside_protected_trees(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    recovery = tmp_path / "recovery"
    reviews = tmp_path / "reviews"
    source.mkdir()
    recovery.mkdir()
    reviews.mkdir()
    path = reviews / "review.json"
    review._write_review_json(path, {"valid": True}, [source, recovery])
    assert json.loads(path.read_text()) == {"valid": True}
    with pytest.raises(review.ReviewError, match="already exists"):
        review._write_review_json(path, {"valid": True}, [source, recovery])
    with pytest.raises(review.ReviewError, match="outside"):
        review._write_review_json(
            recovery / "review.json", {"valid": True}, [source, recovery]
        )


def test_public_review_is_read_only_and_rejects_nonfrozen_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _synthetic_publication(tmp_path, monkeypatch)
    source_before = review._tree_fingerprint(review.SOURCE_STAGING)
    output_before = review._tree_fingerprint(output)
    result = review.review_recovery(output=output)
    assert result["valid"] is False
    assert result["read_only"] is True
    assert any("frozen production output" in error for error in result["errors"])
    assert review._tree_fingerprint(review.SOURCE_STAGING) == source_before
    assert review._tree_fingerprint(output) == output_before
