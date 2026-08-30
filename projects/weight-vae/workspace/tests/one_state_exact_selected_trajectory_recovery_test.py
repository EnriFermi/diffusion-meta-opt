from __future__ import annotations

import ast
import copy
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts import finalize_one_state_exact_selected_trajectory_recovery as recovery


SOURCE_FILE_NAMES = [
    "INCOMPLETE",
    "arm_selection.csv",
    "executed_source_snapshot.py",
    "frozen_dependency_manifest_snapshot.json",
    "intervention_origin.json",
    "intervention_preflight.json",
    "line_search.csv",
    "progress_checkpoint.pt",
    "proposal_diagnostics.csv",
    "protocol_snapshot.md",
    "resolved_config.json",
    "run.log",
    "state_metrics.csv",
    "state_spectra.csv",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_frozen_input(tmp_path: Path) -> tuple[recovery.FrozenInputGuard, Path]:
    root = tmp_path / "repo"
    source = root / "source.incomplete"
    scripts = root / "scripts"
    source.mkdir(parents=True)
    scripts.mkdir()
    producer_source = recovery.ROOT / (
        "scripts/run_one_state_exact_selected_trajectory_continuation.py"
    )
    reviewer_source = recovery.ROOT / (
        "scripts/review_one_state_exact_selected_trajectory_continuation.py"
    )
    producer_copy = scripts / producer_source.name
    reviewer_copy = scripts / reviewer_source.name
    producer_copy.write_bytes(producer_source.read_bytes())
    reviewer_copy.write_bytes(reviewer_source.read_bytes())
    for name in SOURCE_FILE_NAMES:
        path = source / name
        if name == "executed_source_snapshot.py":
            path.write_bytes(producer_source.read_bytes())
        else:
            path.write_text(f"frozen:{name}\n", encoding="utf-8")
    files_sha256 = {name: _sha256(source / name) for name in SOURCE_FILE_NAMES}
    manifest = {
        "protocol_id": recovery.PROTOCOL_ID,
        "source_staging_relative_path": "source.incomplete",
        "expected_exact_file_set": SOURCE_FILE_NAMES,
        "files_sha256": files_sha256,
        "expected_progress": {},
        "expected_original_audit": {
            "failed_gates": ["continuation_diagnostics_recompute"],
            "pass": False,
        },
        "recovery_dependencies": {
            "live_producer_source": {
                "relative_path": f"scripts/{producer_source.name}",
                "sha256": recovery.EXPECTED_PRODUCER_SHA256,
            },
            "independent_continuation_reviewer": {
                "relative_path": f"scripts/{reviewer_source.name}",
                "sha256": recovery.EXPECTED_REVIEWER_SHA256,
            },
        },
        "recovery_radius_rule": {
            "absolute_tolerance": recovery.RECOVERY_RADIUS_ATOL,
            "relative_tolerance": 0.0,
            "expected_max_absolute_error": 0.0,
            "parent_reviewer_source_relative_path": f"scripts/{reviewer_source.name}",
            "parent_reviewer_source_sha256": recovery.EXPECTED_REVIEWER_SHA256,
        },
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    guard = recovery.FrozenInputGuard.create(
        repository_root=root,
        manifest_path=manifest_path,
        expected_manifest_sha256=_sha256(manifest_path),
    )
    return guard, source


def _tiny_progress() -> dict[str, Any]:
    active = {
        "decoder.a": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "decoder.b": torch.tensor([[3.0]], dtype=torch.float64),
    }
    active_hash, _ = recovery._named_tensor_hash(active)
    state = {"accepted_update": 100, "parameter_hash": active_hash, "metric": 1.0}
    return {
        "protocol_id": recovery.SOURCE_PROTOCOL_ID,
        "parent_checkpoint_sha256": recovery.EXPECTED_PARENT_FINAL_SHA256,
        "parent_progress_checkpoint_sha256": (recovery.EXPECTED_PARENT_PROGRESS_SHA256),
        "selected_arm": "low",
        "accepted_updates": 100,
        "new_accepted_updates": 83,
        "termination": "max_updates_reached",
        "transition_chain_sha256": recovery.EXPECTED_TRANSITION_CHAIN_SHA256,
        "active_parameter_hash": active_hash,
        "active_model_state": active,
        "state_rows": [state],
    }


def _audit_tiny_checkpoint(
    checkpoint: dict[str, Any], progress: dict[str, Any]
) -> dict[str, Any]:
    active_hash, parameter_count = recovery._named_tensor_hash(
        progress["active_model_state"]
    )
    return recovery._audit_checkpoint_lineage(
        checkpoint,
        progress,
        expected_tensor_count=2,
        expected_parameter_count=parameter_count,
        expected_active_hash=active_hash,
    )


class _ThirtyOneParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = torch.nn.ParameterDict(
            {
                f"p{index:02d}": torch.nn.Parameter(torch.tensor(float(index)))
                for index in range(recovery.ACTIVE_TENSOR_COUNT)
            }
        )
        self.register_buffer("counter", torch.tensor([7], dtype=torch.int64))


def _synthetic_geometry_payload() -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    eig = torch.linspace(0.01, 2.0, recovery.DIMENSION, dtype=torch.float64)
    hessian = torch.diag(eig.sqrt())
    matrix = hessian @ hessian.T
    low_mask = eig < recovery.GEOMETRY_LOW_THRESHOLD
    low_basis = torch.eye(recovery.DIMENSION, dtype=torch.float64)[:, low_mask]
    low_projector = low_basis @ low_basis.T
    metrics: dict[str, Any] = recovery._recompute_geometry_metric_closures(
        hessian=hessian,
        matrix=matrix,
        eig=eig,
        low_basis=low_basis,
        low_projector=low_projector,
    )
    metrics.update(
        {
            "task_loss": 1.25,
            "hessian_sec": 0.5,
            "low_basis_hash": recovery._sha256_tensor(low_projector),
        }
    )
    tensors = {
        "hessian": hessian,
        "matrix": matrix,
        "eig": eig,
        "current_low_basis": low_basis,
        "current_low_projector": low_projector,
    }
    payload = {
        "protocol_id": recovery.PROTOCOL_ID,
        "source_protocol_id": recovery.SOURCE_PROTOCOL_ID,
        "source_progress_checkpoint_sha256": recovery.EXPECTED_PROGRESS_SHA256,
        "accepted_vae_checkpoint_sha256": (
            recovery.EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256
        ),
        "active_parameter_hash": recovery.EXPECTED_ACTIVE_PARAMETER_HASH,
        "source_weight_index": recovery.SOURCE_WEIGHT_INDEX,
        "task_name": recovery.EXPECTED_TASK_NAME,
        "tau": recovery.EXPECTED_TASK_TAU,
        "z_sha256": recovery.EXPECTED_Z_SHA256,
        "accepted_update": recovery.MAX_ACCEPTED_UPDATES,
        "geometry_constants": {
            "dimension": recovery.DIMENSION,
            "epsilon": recovery.GEOMETRY_EPSILON,
            "old_beta": recovery.GEOMETRY_OLD_BETA,
            "low_threshold": recovery.GEOMETRY_LOW_THRESHOLD,
            "absolute_tolerance": recovery.FINAL_REPLAY_ATOL,
        },
        **tensors,
        "metrics": metrics,
        "tensor_fingerprints": {
            name: recovery._tensor_fingerprint(value) for name, value in tensors.items()
        },
        "installed_model_aggregate_sha256": "a" * 64,
        "post_replay_model_aggregate_sha256": "a" * 64,
    }
    final_row = dict(metrics)
    final_row.update(
        {
            "accepted_update": recovery.MAX_ACCEPTED_UPDATES,
            "parameter_hash": recovery.EXPECTED_ACTIVE_PARAMETER_HASH,
        }
    )
    return payload, final_row, eig.numpy()


def _synthetic_reconstruction_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.SimpleNamespace, Path]:
    root = tmp_path / "repo"
    run_dir = root / "accepted-run"
    run_dir.mkdir(parents=True)
    contents = {
        "config.json": b'{"config": true}\n',
        "weight_pool.pt": b"synthetic weight pool\n",
        "weight_pool_records.csv": b"task_name,tau\nfashion_mnist,1.0\n",
        "vae_checkpoint.pt": b"synthetic checkpoint\n",
    }
    for name, value in contents.items():
        (run_dir / name).write_bytes(value)
    hashes = {name: _sha256(run_dir / name) for name in contents}
    monkeypatch.setattr(recovery, "ROOT", root)
    monkeypatch.setattr(recovery, "EXPECTED_ACCEPTED_RUN_DIR", run_dir)
    monkeypatch.setattr(
        recovery, "EXPECTED_ACCEPTED_RUN_CONFIG_SHA256", hashes["config.json"]
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ACCEPTED_RUN_WEIGHT_POOL_SHA256",
        hashes["weight_pool.pt"],
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ACCEPTED_RUN_RECORDS_SHA256",
        hashes["weight_pool_records.csv"],
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ACCEPTED_RUN_CHECKPOINT_SHA256",
        hashes["vae_checkpoint.pt"],
    )
    producer = types.SimpleNamespace(
        DEFAULT_RUN_DIR=run_dir,
        EXPECTED_CHECKPOINT=hashes["vae_checkpoint.pt"],
    )
    return producer, run_dir


def test_module_import_does_not_preimport_frozen_producer_or_reviewer() -> None:
    command = (
        "import sys; "
        "import scripts.finalize_one_state_exact_selected_trajectory_recovery; "
        "assert 'scripts.run_one_state_exact_selected_trajectory_continuation' "
        "not in sys.modules; "
        "assert 'scripts.review_one_state_exact_selected_trajectory_continuation' "
        "not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=recovery.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_reconstruction_input_audits_are_immediately_around_load_run() -> None:
    source = inspect.getsource(recovery._fresh_final_geometry_replay)
    pre = source.index(
        "reconstruction_inputs_pre = _audit_reconstruction_input_files(producer)"
    )
    load = source.index(
        "run = producer._load_run(producer.DEFAULT_RUN_DIR, device=device)"
    )
    post = source.index(
        "reconstruction_inputs_post = _audit_reconstruction_input_files(producer)"
    )
    assert pre < load < post
    assert (
        source[pre:load]
        .strip()
        .endswith(
            "reconstruction_inputs_pre = _audit_reconstruction_input_files(producer)"
        )
    )
    assert (
        source[load:post]
        .strip()
        .endswith("run = producer._load_run(producer.DEFAULT_RUN_DIR, device=device)")
    )
    assert '"reconstruction_input_files": reconstruction_input_files' in source
    packet_source = inspect.getsource(recovery._write_recovery_packet)
    assert '"reconstruction_inputs_exact_pre_post_load_run"' in packet_source
    assert '"reconstruction_input_files": reconstruction[' in packet_source


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "weight_pool.pt",
        "weight_pool_records.csv",
        "vae_checkpoint.pt",
    ],
)
def test_reconstruction_input_audit_rejects_each_file_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    producer, run_dir = _synthetic_reconstruction_producer(tmp_path, monkeypatch)
    assert recovery._audit_reconstruction_input_files(producer)["pass"] is True
    (run_dir / filename).write_bytes(b"tampered\n")
    with pytest.raises(recovery.RecoveryError, match=filename):
        recovery._audit_reconstruction_input_files(producer)


def test_reconstruction_input_audit_rejects_producer_schema_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, run_dir = _synthetic_reconstruction_producer(tmp_path, monkeypatch)
    producer.DEFAULT_RUN_DIR = str(run_dir)
    with pytest.raises(recovery.RecoveryError, match="path schema"):
        recovery._audit_reconstruction_input_files(producer)
    producer.DEFAULT_RUN_DIR = run_dir.with_name("wrong-run")
    with pytest.raises(recovery.RecoveryError, match="run path mismatch"):
        recovery._audit_reconstruction_input_files(producer)


def test_reconstruction_input_audit_rejects_checkpoint_constant_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _synthetic_reconstruction_producer(tmp_path, monkeypatch)
    producer.EXPECTED_CHECKPOINT = "0" * 64
    with pytest.raises(recovery.RecoveryError, match="checkpoint constant"):
        recovery._audit_reconstruction_input_files(producer)


@pytest.mark.parametrize("mutation", ["hash", "path", "schema"])
def test_reconstruction_input_pre_post_consistency_rejects_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    producer, _ = _synthetic_reconstruction_producer(tmp_path, monkeypatch)
    before = recovery._audit_reconstruction_input_files(producer)
    after = copy.deepcopy(before)
    if mutation == "hash":
        after["files"]["config.json"]["sha256"] = "0" * 64
        match = "schema/path/hash"
    elif mutation == "path":
        after["files"]["config.json"]["path"] = "/wrong/config.json"
        match = "schema/path"
    else:
        after["unexpected"] = True
        match = "schema"
    with pytest.raises(recovery.RecoveryError, match=match):
        recovery._require_reconstruction_input_pre_post_consistency(before, after)


def test_reconstruction_input_pre_post_consistency_accepts_exact_reaudit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _synthetic_reconstruction_producer(tmp_path, monkeypatch)
    before = recovery._audit_reconstruction_input_files(producer)
    after = recovery._audit_reconstruction_input_files(producer)
    audit = recovery._require_reconstruction_input_pre_post_consistency(before, after)
    assert audit["pass"] is True
    assert audit["pre_post_equal"] is True


def test_reconstruction_input_consistency_rejects_equal_fabricated_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _synthetic_reconstruction_producer(tmp_path, monkeypatch)
    fabricated = recovery._audit_reconstruction_input_files(producer)
    fabricated["files"]["config.json"]["sha256"] = "0" * 64
    with pytest.raises(recovery.RecoveryError, match="schema/path/hash"):
        recovery._require_reconstruction_input_pre_post_consistency(
            fabricated,
            copy.deepcopy(fabricated),
        )


def test_finalizer_has_one_explicit_evaluate_call_and_no_optimizer_calls() -> None:
    tree = ast.parse(Path(recovery.__file__).read_text(encoding="utf-8"))
    call_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)
    assert call_names.count("_evaluate") == 1
    assert not ({"backward", "step", "zero_grad"} & set(call_names))
    replay_source = inspect.getsource(recovery._fresh_final_geometry_replay)
    assert "replay_guard.evaluate(evaluate_geometry)" in replay_source
    assert "replay_guard.begin_evaluation" not in replay_source


def test_finalizer_self_freeze_and_copied_snapshots(tmp_path: Path) -> None:
    source_snapshot = tmp_path / "finalizer.py"
    protocol_snapshot = tmp_path / "protocol.md"
    shutil.copyfile(recovery.LIVE_SOURCE, source_snapshot)
    shutil.copyfile(recovery.RECOVERY_PROTOCOL, protocol_snapshot)
    audit = recovery._verify_finalizer_source_freeze(
        snapshot=source_snapshot,
        protocol_snapshot=protocol_snapshot,
    )
    assert (
        audit["live_normalized_source_sha256"]
        == recovery.EXPECTED_NORMALIZED_SOURCE_SHA256
    )
    assert audit["snapshot_source_sha256"] == audit["live_source_sha256"]


def test_finalizer_self_freeze_rejects_snapshot_tampering(tmp_path: Path) -> None:
    source_snapshot = tmp_path / "finalizer.py"
    protocol_snapshot = tmp_path / "protocol.md"
    shutil.copyfile(recovery.LIVE_SOURCE, source_snapshot)
    shutil.copyfile(recovery.RECOVERY_PROTOCOL, protocol_snapshot)
    source_snapshot.write_text(
        source_snapshot.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(recovery.RecoveryError, match="source identity"):
        recovery._verify_finalizer_source_freeze(
            snapshot=source_snapshot,
            protocol_snapshot=protocol_snapshot,
        )


def test_finalizer_self_freeze_rejects_protocol_snapshot_tampering(
    tmp_path: Path,
) -> None:
    source_snapshot = tmp_path / "finalizer.py"
    protocol_snapshot = tmp_path / "protocol.md"
    shutil.copyfile(recovery.LIVE_SOURCE, source_snapshot)
    protocol_snapshot.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="protocol snapshot"):
        recovery._verify_finalizer_source_freeze(
            snapshot=source_snapshot,
            protocol_snapshot=protocol_snapshot,
        )


def test_recovery_task_and_import_manifests_are_exactly_pinned() -> None:
    task, task_audit = recovery._verify_recovery_task_manifest()
    imports, import_audit = recovery._verify_recovery_import_manifest()
    assert task_audit["manifest_sha256"] == (
        recovery.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256
    )
    assert task["expected_runtime"] == recovery.EXPECTED_RUNTIME
    assert import_audit["module_count"] == 49
    assert set(imports["entry_modules"]) <= set(imports["loaded_repository_modules"])


@pytest.mark.parametrize(
    ("source", "expected_hash", "verifier"),
    [
        (
            recovery.RECOVERY_TASK_MANIFEST,
            recovery.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256,
            recovery._verify_recovery_task_manifest,
        ),
        (
            recovery.RECOVERY_IMPORT_MANIFEST,
            recovery.EXPECTED_RECOVERY_IMPORT_MANIFEST_SHA256,
            recovery._verify_recovery_import_manifest,
        ),
    ],
)
def test_recovery_dependency_manifests_reject_tampering(
    tmp_path: Path,
    source: Path,
    expected_hash: str,
    verifier: Any,
) -> None:
    manifest = tmp_path / source.name
    manifest.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(recovery.RecoveryError, match="manifest hash mismatch"):
        verifier(
            repository_root=tmp_path,
            manifest_path=manifest,
            expected_manifest_sha256=expected_hash,
        )


def test_recovery_task_manifest_rejects_symlinked_task_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    manifest = root / "recovery_task_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(recovery.RECOVERY_TASK_MANIFEST.read_bytes())
    for relative in recovery.EXPECTED_TASK_SOURCE_DEPENDENCIES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((recovery.ROOT / relative).read_bytes())
    target = root / next(iter(recovery.EXPECTED_TASK_SOURCE_DEPENDENCIES))
    backing = target.with_suffix(".backing.py")
    target.rename(backing)
    target.symlink_to(backing)
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        recovery._verify_recovery_task_manifest(
            repository_root=root,
            manifest_path=manifest,
            expected_manifest_sha256=recovery.EXPECTED_RECOVERY_TASK_MANIFEST_SHA256,
        )


def test_runtime_task_reconstruction_checks_all_tensor_bytes_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "vae_checkpoint.pt"
    checkpoint.write_bytes(b"accepted checkpoint")
    task_set = types.SimpleNamespace(
        task_name=recovery.EXPECTED_TASK_NAME,
        train_images=torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2),
        train_labels=torch.tensor([0, 1, 2], dtype=torch.int64),
        test_images=torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2),
        test_labels=torch.tensor([1, 0], dtype=torch.int64),
    )
    expected_tensors = {}
    for name in ("train_images", "train_labels", "test_images", "test_labels"):
        fingerprint = recovery._tensor_fingerprint(getattr(task_set, name))
        fingerprint.pop("numel")
        expected_tensors[name] = fingerprint
    monkeypatch.setattr(recovery, "EXPECTED_TASK_TENSORS", expected_tensors)
    monkeypatch.setattr(recovery, "EXPECTED_TASK_TRAIN_COUNT", 3)
    monkeypatch.setattr(recovery, "EXPECTED_TASK_TEST_COUNT", 2)
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ACCEPTED_VAE_CHECKPOINT_SHA256",
        _sha256(checkpoint),
    )
    manifest = {
        "protocol_id": recovery.PROTOCOL_ID,
        "source_weight_index": recovery.SOURCE_WEIGHT_INDEX,
        "task_name": recovery.EXPECTED_TASK_NAME,
        "tau": recovery.EXPECTED_TASK_TAU,
        "accepted_vae_checkpoint_sha256": _sha256(checkpoint),
        "selected_task_tensors": expected_tensors,
    }
    record = {
        "source_weight_index": recovery.SOURCE_WEIGHT_INDEX,
        "task_name": recovery.EXPECTED_TASK_NAME,
        "tau": recovery.EXPECTED_TASK_TAU,
    }
    audit = recovery._audit_runtime_task_reconstruction(
        task_manifest=manifest,
        record=record,
        task_set=task_set,
        accepted_checkpoint=checkpoint,
    )
    assert audit["pass"] is True
    task_set.test_labels[0] += 1
    with pytest.raises(recovery.RecoveryError, match="runtime task reconstruction"):
        recovery._audit_runtime_task_reconstruction(
            task_manifest=manifest,
            record=record,
            task_set=task_set,
            accepted_checkpoint=checkpoint,
        )


def test_frozen_input_guard_accepts_exact_packet(tmp_path: Path) -> None:
    guard, _ = _synthetic_frozen_input(tmp_path)
    audit = guard.verify()
    assert audit["source_file_count"] == 14
    assert audit["dependency_files_sha256"] == {
        "independent_continuation_reviewer": recovery.EXPECTED_REVIEWER_SHA256,
        "live_producer_source": recovery.EXPECTED_PRODUCER_SHA256,
    }


@pytest.mark.parametrize(
    "tampered_name", ["arm_selection.csv", "progress_checkpoint.pt"]
)
def test_frozen_input_guard_rejects_source_tampering(
    tmp_path: Path, tampered_name: str
) -> None:
    guard, source = _synthetic_frozen_input(tmp_path)
    (source / tampered_name).write_bytes(b"tampered")
    with pytest.raises(recovery.RecoveryError, match="hash mismatch"):
        guard.verify()


def test_frozen_input_guard_rejects_extra_file(tmp_path: Path) -> None:
    guard, source = _synthetic_frozen_input(tmp_path)
    (source / "decision.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="exact file set mismatch"):
        guard.verify()


def test_frozen_input_guard_rejects_dependency_tampering(tmp_path: Path) -> None:
    guard, _ = _synthetic_frozen_input(tmp_path)
    reviewer = guard.dependency_paths()["independent_continuation_reviewer"]
    reviewer.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="dependency hash mismatch"):
        guard.verify()


def test_frozen_input_guard_rejects_source_directory_symlink_clone(
    tmp_path: Path,
) -> None:
    guard, source = _synthetic_frozen_input(tmp_path)
    backing = source.with_name("source.backing")
    source.rename(backing)
    source.symlink_to(backing, target_is_directory=True)
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        guard.verify()


def test_frozen_input_guard_rejects_dependency_file_symlink(tmp_path: Path) -> None:
    guard, _ = _synthetic_frozen_input(tmp_path)
    reviewer = guard.dependency_paths()["independent_continuation_reviewer"]
    backing = reviewer.with_suffix(".backing.py")
    reviewer.rename(backing)
    reviewer.symlink_to(backing)
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        guard.verify()


def test_frozen_input_guard_rejects_intermediate_component_symlink(
    tmp_path: Path,
) -> None:
    guard, _ = _synthetic_frozen_input(tmp_path)
    scripts = guard.repository_root / "scripts"
    backing = guard.repository_root / "scripts.backing"
    scripts.rename(backing)
    scripts.symlink_to(backing, target_is_directory=True)
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        guard.verify()


def test_frozen_input_guard_rejects_symlinked_manifest(tmp_path: Path) -> None:
    guard, _ = _synthetic_frozen_input(tmp_path)
    manifest_link = guard.repository_root / "manifest-link.json"
    manifest_link.symlink_to(guard.manifest_path)
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        recovery.FrozenInputGuard.create(
            repository_root=guard.repository_root,
            manifest_path=manifest_link,
            expected_manifest_sha256=guard.expected_manifest_sha256,
        )


def test_frozen_input_guard_rejects_manifest_mutation(tmp_path: Path) -> None:
    guard, _ = _synthetic_frozen_input(tmp_path)
    guard.manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="manifest changed"):
        guard.verify()


def test_transitive_dependency_manifest_tree_rejects_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    dependency = root / "dependency.py"
    nested_dependency = root / "nested.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    nested_dependency.write_text("VALUE = 2\n", encoding="utf-8")
    nested_manifest = root / "nested_frozen_dependency_manifest.json"
    _write_json(nested_manifest, {"nested.py": _sha256(nested_dependency)})
    manifest = root / "frozen_dependency_manifest.json"
    _write_json(
        manifest,
        {
            "dependency.py": _sha256(dependency),
            nested_manifest.name: _sha256(nested_manifest),
        },
    )
    audit = recovery._verify_dependency_manifest_tree(
        repository_root=root,
        manifest_path=manifest,
        expected_manifest_sha256=_sha256(manifest),
    )
    assert audit["manifest_count"] == 2
    assert audit["verified_file_count"] == 3
    nested_dependency.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="transitive dependency"):
        recovery._verify_dependency_manifest_tree(
            repository_root=root,
            manifest_path=manifest,
            expected_manifest_sha256=_sha256(manifest),
        )


@pytest.mark.parametrize("mode", ["named", "aliased_path"])
def test_preimport_audit_rejects_preloaded_frozen_modules(mode: str) -> None:
    setup = (
        "import importlib; "
        "importlib.import_module("
        "'post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline'"
        ")"
        if mode == "named"
        else (
            "import sys, types; "
            "m=types.ModuleType('injected_frozen_alias'); "
            "m.__file__=str(r.ROOT/'scripts/run_one_state_exact_selected_trajectory_continuation.py'); "
            "sys.modules[m.__name__]=m"
        )
    )
    command = f"""
from scripts import finalize_one_state_exact_selected_trajectory_recovery as r
{setup}
g = r.FrozenInputGuard.create(
    repository_root=r.ROOT,
    manifest_path=r.FROZEN_INPUT_MANIFEST,
    expected_manifest_sha256=r.EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
)
try:
    r._producer_dependency_preimport_audit(g)
except r.RecoveryError as error:
    assert 'preloaded' in str(error)
else:
    raise AssertionError('preloaded frozen module was accepted')
"""
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=recovery.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_clean_subprocess_imports_exact_forty_nine_module_closure() -> None:
    command = """
import sys
from scripts import finalize_one_state_exact_selected_trajectory_recovery as r
sys.modules.pop('__mp_main__', None)
g = r.FrozenInputGuard.create(
    repository_root=r.ROOT,
    manifest_path=r.FROZEN_INPUT_MANIFEST,
    expected_manifest_sha256=r.EXPECTED_FROZEN_INPUT_MANIFEST_SHA256,
)
_, _, audit = r._import_verified_dependencies(g)
assert audit['loaded_repository_closure']['module_count'] == 49
producer = __import__(
    'scripts.run_one_state_exact_selected_trajectory_continuation',
    fromlist=['unused'],
)
inputs = r._audit_reconstruction_input_files(producer)
assert inputs['pass'] is True
assert inputs['file_count'] == 4
"""
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=recovery.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_imported_module_closure_rejected(setup: str, expected_error: str) -> None:
    command = f"""
import sys
import types
from scripts import finalize_one_state_exact_selected_trajectory_recovery as r
{setup}
try:
    r._audit_loaded_repository_module_closure(
        repository_root=r.ROOT,
        import_manifest={{'loaded_repository_modules': {{}}}},
    )
except r.RecoveryError as error:
    assert {expected_error!r} in str(error), str(error)
else:
    raise AssertionError('invalid repository module alias was accepted')
"""
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=recovery.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_script_namespace_same_object_mp_main_alias_passes_closure() -> None:
    command = """
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
script_main = sys.modules['__main__']
script_main.__dict__['__file__'] = str(source)
script_main.__dict__['__name__'] = '__closure_probe__'
sys.modules['__closure_probe__'] = script_main
exec(compile(source.read_bytes(), str(source), 'exec'), script_main.__dict__)
del sys.modules['__closure_probe__']
script_main.__dict__['__name__'] = '__main__'
sys.modules['__mp_main__'] = script_main

assert sys.modules['__mp_main__'] is sys.modules['__main__']
assert script_main.__dict__ is _audit_loaded_repository_module_closure.__globals__
assert Path(script_main.__file__).resolve() == LIVE_SOURCE
audit = _audit_loaded_repository_module_closure(
    repository_root=ROOT,
    import_manifest={'loaded_repository_modules': {}},
)
assert audit == {
    'pass': True,
    'module_count': 0,
    'modules': {},
    'allowed_additions': [
        '__main__',
        'scripts.finalize_one_state_exact_selected_trajectory_recovery',
        '__mp_main__',
    ],
}
"""
    subprocess.run(
        [sys.executable, "-c", command, str(recovery.LIVE_SOURCE)],
        cwd=recovery.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_mp_main_different_object_with_live_file_is_rejected() -> None:
    _assert_imported_module_closure_rejected(
        """
sys.modules['__main__'] = r
fake = types.ModuleType('__mp_main__')
fake.__file__ = str(r.LIVE_SOURCE)
sys.modules['__mp_main__'] = fake
""",
        "invalid __mp_main__ repository closure alias: __main__ identity mismatch",
    )


def test_fake_module_bound_to_main_and_mp_main_with_live_file_is_rejected() -> None:
    _assert_imported_module_closure_rejected(
        """
fake = types.ModuleType('__main__')
fake.__file__ = str(r.LIVE_SOURCE)
sys.modules['__main__'] = fake
sys.modules['__mp_main__'] = fake
""",
        "invalid __mp_main__ repository closure alias: "
        "finalizer globals provenance mismatch",
    )


def test_same_object_mp_main_alias_with_wrong_file_is_rejected() -> None:
    _assert_imported_module_closure_rejected(
        """
r.__file__ = str(r.ROOT / 'tests/one_state_exact_selected_trajectory_recovery_test.py')
sys.modules['__main__'] = r
sys.modules['__mp_main__'] = r
""",
        "invalid __mp_main__ repository closure alias: "
        "__main__.__file__ does not resolve exactly to LIVE_SOURCE",
    )


@pytest.mark.parametrize(
    "setup",
    [
        "sys.modules['__mp_main__'] = None",
        "sys.modules['__mp_main__'] = object()",
        """
r.__file__ = 'scripts/finalize_one_state_exact_selected_trajectory_recovery.py'
sys.modules['__main__'] = r
sys.modules['__mp_main__'] = r
""",
        """
r.__file__ = '<synthetic-finalizer>'
sys.modules['__main__'] = r
sys.modules['__mp_main__'] = r
""",
    ],
    ids=["none", "nonmodule", "relative-file", "synthetic-file"],
)
def test_malformed_mp_main_entries_are_rejected_before_generic_skip(
    setup: str,
) -> None:
    _assert_imported_module_closure_rejected(
        setup,
        "invalid __mp_main__ repository closure alias",
    )


def test_arbitrary_live_finalizer_alias_remains_rejected() -> None:
    _assert_imported_module_closure_rejected(
        """
sys.modules.pop('__mp_main__', None)
sys.modules['arbitrary_finalizer_alias'] = r
""",
        "extras={'arbitrary_finalizer_alias': "
        "'scripts/finalize_one_state_exact_selected_trajectory_recovery.py'}",
    )


def test_output_must_be_disjoint_from_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(recovery.RecoveryError, match="disjoint"):
        recovery._validate_output_separation(source, source / "child")
    with pytest.raises(recovery.RecoveryError, match="disjoint"):
        recovery._validate_output_separation(source, tmp_path)


def test_public_finalizer_and_cli_have_no_output_or_device_override(
    tmp_path: Path,
) -> None:
    assert list(inspect.signature(recovery.finalize_recovery).parameters) == []
    assert recovery._require_default_output(recovery.DEFAULT_OUTPUT) == (
        recovery.DEFAULT_OUTPUT
    )
    with pytest.raises(recovery.RecoveryError, match="frozen"):
        recovery._require_default_output(tmp_path / "arbitrary-output")
    for flag in ("--output", "--device"):
        result = subprocess.run(
            [sys.executable, str(recovery.LIVE_SOURCE), flag, "forbidden"],
            cwd=recovery.ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr


def test_production_device_selector_is_exact_cuda_zero() -> None:
    assert recovery.PRODUCTION_DEVICE == "cuda:0"


def test_original_audit_requires_exact_frozen_failed_set() -> None:
    def expected_failure() -> None:
        raise RuntimeError(
            "continuation progress audit failed: ['continuation_diagnostics_recompute']"
        )

    result = recovery._require_original_audit_failure(
        expected_failure,
        expected_failed_gates=["continuation_diagnostics_recompute"],
    )
    assert result["pass"] is False
    assert result["audit_invocations"] == 1


@pytest.mark.parametrize(
    "message",
    [
        "continuation progress audit failed: ['another_gate']",
        "continuation progress audit failed: "
        "['continuation_diagnostics_recompute', 'another_gate']",
        "unrelated failure",
    ],
)
def test_original_audit_rejects_wrong_failed_set(message: str) -> None:
    def wrong_failure() -> None:
        raise RuntimeError(message)

    with pytest.raises(recovery.RecoveryError):
        recovery._require_original_audit_failure(
            wrong_failure,
            expected_failed_gates=["continuation_diagnostics_recompute"],
        )


def test_original_audit_rejects_unexpected_pass() -> None:
    with pytest.raises(recovery.RecoveryError, match="unexpectedly passed"):
        recovery._require_original_audit_failure(
            lambda: None,
            expected_failed_gates=["continuation_diagnostics_recompute"],
        )


def test_radius_audit_accepts_frozen_tolerance_and_reports_old_violations() -> None:
    rows = [
        {"target_update": 18, "direction_norm": 1.0 + 2e-9},
        {"target_update": 19, "direction_norm": 1.0 - 4e-9},
    ]
    audit = recovery._radius_audit(
        rows,
        target_norm=1.0,
        absolute_tolerance=5e-9,
    )
    assert audit["pass"] is True
    assert audit["count_exceeding_original_1e_9"] == 2
    assert audit["count_exceeding_recovery_5e_9"] == 0
    assert audit["max_error_target_update"] == 19


def test_radius_audit_rejects_error_above_five_e_minus_nine() -> None:
    rows = [{"target_update": 18, "direction_norm": 1.0 + 5.1e-9}]
    with pytest.raises(recovery.RecoveryError, match="exceeds"):
        recovery._radius_audit(
            rows,
            target_norm=1.0,
            absolute_tolerance=5e-9,
        )


def test_checkpoint_is_bitwise_derived_from_terminal_progress(tmp_path: Path) -> None:
    progress = _tiny_progress()
    checkpoint = recovery._build_final_checkpoint(
        progress=progress,
        source_progress_sha256=recovery.EXPECTED_PROGRESS_SHA256,
    )
    path = tmp_path / "final_checkpoint.pt"
    recovery._atomic_torch_save(path, checkpoint)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    active_hash, parameter_count = recovery._named_tensor_hash(
        progress["active_model_state"]
    )
    audit = recovery._audit_checkpoint_lineage(
        reloaded,
        progress,
        expected_tensor_count=2,
        expected_parameter_count=parameter_count,
        expected_active_hash=active_hash,
    )
    assert audit["pass"] is True
    assert audit["bitwise_tensor_equality"] is True
    for name, source in progress["active_model_state"].items():
        assert torch.equal(reloaded["active_model_state"][name], source)
        assert reloaded["active_model_state"][name].data_ptr() != source.data_ptr()


def test_checkpoint_lineage_rejects_tensor_mutation() -> None:
    progress = _tiny_progress()
    checkpoint = recovery._build_final_checkpoint(
        progress=progress,
        source_progress_sha256=recovery.EXPECTED_PROGRESS_SHA256,
    )
    checkpoint["active_model_state"]["decoder.a"][0] += 1.0
    active_hash, parameter_count = recovery._named_tensor_hash(
        progress["active_model_state"]
    )
    with pytest.raises(recovery.RecoveryError, match="bitwise"):
        recovery._audit_checkpoint_lineage(
            checkpoint,
            progress,
            expected_tensor_count=2,
            expected_parameter_count=parameter_count,
            expected_active_hash=active_hash,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("protocol_id", "wrong"),
        ("source_protocol_id", "wrong"),
        ("source_progress_checkpoint_sha256", "0" * 64),
        ("parent_checkpoint_sha256", "0" * 64),
        ("parent_progress_checkpoint_sha256", "0" * 64),
        ("selected_arm", "high"),
        ("accepted_updates", 99),
        ("new_accepted_updates", 82),
        ("termination", "wrong"),
        ("transition_chain_sha256", "0" * 64),
        ("active_parameter_hash", "0" * 64),
        ("source_weight_index", 379),
        ("z_sha256", "0" * 64),
        ("accepted_vae_checkpoint_sha256", "0" * 64),
        ("stored_state_metrics", {"wrong": True}),
    ],
)
def test_checkpoint_lineage_rejects_every_metadata_claim(
    field: str, bad_value: Any
) -> None:
    progress = _tiny_progress()
    checkpoint = recovery._build_final_checkpoint(
        progress=progress,
        source_progress_sha256=recovery.EXPECTED_PROGRESS_SHA256,
    )
    checkpoint[field] = bad_value
    with pytest.raises(recovery.RecoveryError, match="metadata/hash"):
        _audit_tiny_checkpoint(checkpoint, progress)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_checkpoint_lineage_rejects_nonexact_schema(mutation: str) -> None:
    progress = _tiny_progress()
    checkpoint = recovery._build_final_checkpoint(
        progress=progress,
        source_progress_sha256=recovery.EXPECTED_PROGRESS_SHA256,
    )
    if mutation == "missing":
        checkpoint.pop("z_sha256")
    else:
        checkpoint["unexpected"] = True
    with pytest.raises(recovery.RecoveryError, match="exact schema"):
        _audit_tiny_checkpoint(checkpoint, progress)


def test_geometry_guard_allows_exactly_one_evaluation() -> None:
    guard = recovery.GeometryReplayGuard()
    calls: list[int] = []

    def evaluate(value: int) -> int:
        calls.append(value)
        return value + 1

    assert guard.evaluate(evaluate, 4) == 5
    execution = guard.require_complete()
    assert execution["geometry_evaluation_count"] == 1
    assert execution["optimization_gradient_evaluations"] == 0
    assert calls == [4]
    with pytest.raises(recovery.RecoveryError, match="budget exceeded"):
        guard.evaluate(evaluate, 5)
    assert calls == [4]


def test_geometry_guard_fails_closed_on_gradient_or_update_attempt() -> None:
    for operation in (
        "optimization_gradient",
        "proposal",
        "line_search",
        "parameter_update",
    ):
        guard = recovery.GeometryReplayGuard()
        with pytest.raises(recovery.RecoveryError, match=operation):
            guard.forbid(operation)
        assert guard.forbidden_operation_attempts == 1
        with pytest.raises(recovery.RecoveryError, match="invariant"):
            guard.require_complete()


def test_model_installation_changes_exactly_thirty_one_active_parameters() -> None:
    model = _ThirtyOneParameterModel()
    base = recovery._snapshot_model_tensors(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    installed = recovery._snapshot_model_tensors(model)
    audit = recovery._audit_model_installation(
        base=base,
        installed=installed,
        expected_changed_parameters=set(dict(model.named_parameters())),
    )
    assert audit["changed_parameter_count"] == recovery.ACTIVE_TENSOR_COUNT
    assert audit["changed_buffer_count"] == 0


def test_model_installation_rejects_buffer_or_inactive_mutation() -> None:
    model = _ThirtyOneParameterModel()
    base = recovery._snapshot_model_tensors(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
        model.counter.add_(1)
    installed = recovery._snapshot_model_tensors(model)
    with pytest.raises(recovery.RecoveryError, match="unexpected model tensor"):
        recovery._audit_model_installation(
            base=base,
            installed=installed,
            expected_changed_parameters=set(dict(model.named_parameters())),
        )


def test_model_replay_nonmutation_is_bitwise_for_all_parameters_and_buffers() -> None:
    model = _ThirtyOneParameterModel()
    installed = recovery._snapshot_model_tensors(model)
    replayed = recovery._snapshot_model_tensors(model)
    audit = recovery._audit_model_replay_nonmutation(
        installed=installed,
        replayed=replayed,
    )
    assert audit["all_parameters_bitwise_unchanged"] is True
    assert audit["all_buffers_bitwise_unchanged"] is True
    with torch.no_grad():
        model.values["p00"].add_(1.0)
    with pytest.raises(recovery.RecoveryError, match="mutated model tensors"):
        recovery._audit_model_replay_nonmutation(
            installed=installed,
            replayed=recovery._snapshot_model_tensors(model),
        )


def _write_spectrum_csv(path: Path, stored: np.ndarray, replayed: np.ndarray) -> None:
    recovery._atomic_csv(
        path,
        [
            {
                "rank": rank,
                "stored": float(stored[rank]),
                "replayed": float(replayed[rank]),
                "abs_error": abs(float(replayed[rank]) - float(stored[rank])),
            }
            for rank in range(recovery.DIMENSION)
        ],
    )


def test_replayed_spectrum_csv_contains_and_verifies_all_512_values(
    tmp_path: Path,
) -> None:
    stored = np.linspace(0.0, 2.0, recovery.DIMENSION, dtype=np.float64)
    replayed = stored.copy()
    replayed[123] += 4e-10
    path = tmp_path / "replayed_final_spectrum.csv"
    _write_spectrum_csv(path, stored, replayed)
    audit = recovery._audit_replayed_final_spectrum_file(
        path=path,
        stored_eig=stored,
        replayed_eig=replayed,
    )
    assert audit["row_count"] == recovery.DIMENSION
    assert audit["max_error_rank"] == 123


@pytest.mark.parametrize("field", ["rank", "stored", "replayed", "abs_error"])
def test_replayed_spectrum_csv_rejects_each_tampered_field(
    tmp_path: Path, field: str
) -> None:
    stored = np.linspace(0.0, 2.0, recovery.DIMENSION, dtype=np.float64)
    replayed = stored.copy()
    path = tmp_path / "replayed_final_spectrum.csv"
    _write_spectrum_csv(path, stored, replayed)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[17][field] = "99" if field == "rank" else "1.25"
    recovery._atomic_csv(path, rows)
    with pytest.raises(recovery.RecoveryError, match="spectrum CSV"):
        recovery._audit_replayed_final_spectrum_file(
            path=path,
            stored_eig=stored,
            replayed_eig=replayed,
        )


def test_replayed_spectrum_csv_rejects_error_above_tolerance(
    tmp_path: Path,
) -> None:
    stored = np.zeros(recovery.DIMENSION, dtype=np.float64)
    replayed = stored.copy()
    replayed[0] = 1.1e-9
    path = tmp_path / "replayed_final_spectrum.csv"
    _write_spectrum_csv(path, stored, replayed)
    with pytest.raises(recovery.RecoveryError, match="exceeds tolerance"):
        recovery._audit_replayed_final_spectrum_file(
            path=path,
            stored_eig=stored,
            replayed_eig=replayed,
        )


def test_full_replayed_geometry_payload_has_independent_scientific_closures() -> None:
    payload, final_row, stored_eig = _synthetic_geometry_payload()
    audit = recovery._audit_replayed_geometry_payload(
        payload=payload,
        final_row=final_row,
        stored_eig=stored_eig,
    )
    assert audit["pass"] is True
    assert audit["matrix_from_hessian_max_abs_error"] <= 1e-12
    assert audit["eigvalsh_matrix_max_abs_error"] <= 1e-12
    assert audit["low_projector_basis_max_abs_error"] <= 1e-12


@pytest.mark.parametrize(
    "mutation",
    ["matrix", "eig", "projector", "checkpoint", "model_nonmutation"],
)
def test_full_replayed_geometry_payload_rejects_tampering(mutation: str) -> None:
    payload, final_row, stored_eig = _synthetic_geometry_payload()
    payload = copy.deepcopy(payload)
    if mutation in {"matrix", "eig", "projector"}:
        key = {
            "matrix": "matrix",
            "eig": "eig",
            "projector": "current_low_projector",
        }[mutation]
        payload[key].reshape(-1)[0] += 1e-6
        payload["tensor_fingerprints"][key] = recovery._tensor_fingerprint(payload[key])
    elif mutation == "checkpoint":
        payload["accepted_vae_checkpoint_sha256"] = "0" * 64
    else:
        payload["post_replay_model_aggregate_sha256"] = "b" * 64
    with pytest.raises(recovery.RecoveryError):
        recovery._audit_replayed_geometry_payload(
            payload=payload,
            final_row=final_row,
            stored_eig=stored_eig,
        )


def test_exact_staged_entry_set_accepts_only_regular_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    audit = recovery._validate_exact_regular_file_set(tmp_path, {"a.txt"})
    assert audit["entry_count"] == 1
    (tmp_path / "extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="exact entry set"):
        recovery._validate_exact_regular_file_set(tmp_path, {"a.txt"})


@pytest.mark.parametrize("entry_kind", ["directory", "symlink", "fifo"])
def test_exact_staged_entry_set_rejects_nonregular_entries(
    tmp_path: Path, entry_kind: str
) -> None:
    entry = tmp_path / "entry"
    if entry_kind == "directory":
        entry.mkdir()
    elif entry_kind == "symlink":
        target = tmp_path.parent / f"{tmp_path.name}-target"
        target.write_text("target\n", encoding="utf-8")
        entry.symlink_to(target)
    else:
        os.mkfifo(entry)
    with pytest.raises(recovery.RecoveryError):
        recovery._validate_exact_regular_file_set(tmp_path, {"entry"})


def test_atomic_publication_runs_prepublish_before_exposure(tmp_path: Path) -> None:
    output = tmp_path / "published"
    observations: list[str] = []

    def build(stage: Path) -> str:
        _write_json(stage / "FINALIZED.json", {"valid": True})
        return "payload"

    def prepublish(stage: Path) -> None:
        assert not output.exists()
        assert (stage / "INCOMPLETE").is_file()
        assert (stage / "FINALIZED.json").is_file()
        observations.append("prepublish")

    result = recovery._atomic_directory_publication(
        output,
        build,
        prepublish=prepublish,
    )
    assert observations == ["prepublish"]
    assert result.payload == "payload"
    assert output.is_dir()
    assert not (output / "INCOMPLETE").exists()


def test_atomic_publication_preserves_failed_prepublish_staging(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"

    def build(stage: Path) -> None:
        _write_json(stage / "FINALIZED.json", {"valid": True})

    def reject(_: Path) -> None:
        raise recovery.RecoveryError("injected prepublish rejection")

    with pytest.raises(recovery.RecoveryError, match="injected") as caught:
        recovery._atomic_directory_publication(
            output,
            build,
            prepublish=reject,
        )
    failed = caught.value.failed_recovery_staging
    assert not output.exists()
    assert failed.is_dir()
    assert (failed / "INCOMPLETE").is_file()
    assert (failed / "failure.json").is_file()


def test_failed_publication_never_follows_tampered_incomplete_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")

    def fail_with_symlink(stage: Path) -> None:
        (stage / "INCOMPLETE").unlink()
        (stage / "INCOMPLETE").symlink_to(sentinel)
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError) as caught:
        recovery._atomic_directory_publication(output, fail_with_symlink)
    failed = caught.value.failed_recovery_staging
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"
    assert (failed / "INCOMPLETE").is_file()
    assert not (failed / "INCOMPLETE").is_symlink()


def test_atomic_publication_exposes_only_complete_directory(tmp_path: Path) -> None:
    output = tmp_path / "published"

    def build(stage: Path) -> str:
        assert not output.exists()
        assert (stage / "INCOMPLETE").is_file()
        _write_json(stage / "FINALIZED.json", {"valid": True})
        (stage / "payload.txt").write_text("complete\n", encoding="utf-8")
        return "done"

    result = recovery._atomic_directory_publication(output, build)
    assert result.output == output.resolve()
    assert result.payload == "done"
    assert result.failed_staging is None
    assert output.is_dir()
    assert not (output / "INCOMPLETE").exists()
    assert (output / "payload.txt").read_text() == "complete\n"
    assert not list(tmp_path.glob(".*.incomplete.*"))


def test_failed_atomic_publication_is_preserved_separately(tmp_path: Path) -> None:
    output = tmp_path / "published"

    def fail(stage: Path) -> None:
        (stage / "partial.txt").write_text("partial\n", encoding="utf-8")
        (stage / "INCOMPLETE").unlink()
        raise ValueError("injected failure")

    with pytest.raises(ValueError, match="injected failure") as caught:
        recovery._atomic_directory_publication(output, fail)
    assert not output.exists()
    failed = getattr(caught.value, "failed_recovery_staging")
    assert isinstance(failed, Path) and failed.is_dir()
    assert failed.parent == output.parent
    assert failed.name.startswith("published.failed.")
    assert (failed / "INCOMPLETE").is_file()
    assert (failed / "partial.txt").read_text() == "partial\n"
    failure = json.loads((failed / "failure.json").read_text())
    assert failure["status"] == "failed_recovery_staging"
    assert failure["error_type"] == "ValueError"


def test_atomic_publication_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        recovery._atomic_directory_publication(output, lambda _: None)
