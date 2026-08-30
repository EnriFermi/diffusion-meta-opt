#!/usr/bin/env python3
"""Build deduplicated activation/context and weight-tile banks from the zoo."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from big_vae.weightclip_benchmark.activation_capture import (  # noqa: E402
    capture_native_train_activations,
    context_tiles_batched,
    stable_seed,
)
from big_vae.weightclip_benchmark.dataset_payload import (  # noqa: E402
    load_cached_dataset_payload,
)
from big_vae.datasets.operator_bank import validate_array_bank_contents  # noqa: E402
from big_vae.weightclip_benchmark.manifests import (  # noqa: E402
    ImmutableArrayShardWriter,
    assert_file_snapshot,
    operator_bank_builder_source_seal,
    sha256_file,
    sha256_file_stable,
    validate_operator_bank_builder_source_seal,
    validate_operator_checkpoint_selection,
    write_json_atomic,
    write_json_immutable,
    write_records_immutable,
)
from big_vae.weightclip_benchmark.metadata import (  # noqa: E402
    OperatorDatasetProtocol,
    SOURCE_TASKS,
    canonical_json_bytes,
    redact_secrets,
)
from big_vae.weightclip_benchmark.parameter_adapters import (  # noqa: E402
    apply_resnet_gauge,
    assert_complete_coverage,
    gauge_id,
    invert_resnet_gauge,
    parameter_to_matrix,
    sample_resnet_gauge,
    supported_operator_specs,
    tile_matrix,
)
from big_vae.weightclip_benchmark.resnet18slim import ResNet18Slim  # noqa: E402


@contextmanager
def _exclusive_operator_bank_build_lock(bank_root: Path) -> Iterator[Path]:
    """Hold one crash-releasing nonblocking lock over preflight through commit."""

    resolved_root = bank_root.expanduser().resolve()
    resolved_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_root.parent / f".{resolved_root.name}.operator-bank-build.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another operator-bank builder holds the lock: {lock_path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\nroot={resolved_root}\n".encode())
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _assert_snapshot_around_load(
    snapshot: dict[str, Any],
    loader: Any,
    *,
    context: str,
) -> Any:
    assert_file_snapshot(snapshot, context=f"{context} before load")
    value = loader(Path(str(snapshot["path"])))
    assert_file_snapshot(snapshot, context=f"{context} after load")
    return value


def _assert_build_input_snapshots(
    *,
    checkpoint_manifest: dict[str, Any],
    config_source: dict[str, Any] | None,
    activation_datasets: Iterable[dict[str, Any]],
    checkpoint_files: Iterable[dict[str, Any]],
    builder_source_snapshots: Iterable[dict[str, Any]],
    builder_source_seal: dict[str, Any],
) -> None:
    """Final no-write barrier immediately before immutable bank publication."""

    assert_file_snapshot(checkpoint_manifest, context="checkpoint manifest post-build")
    if config_source is not None:
        assert_file_snapshot(config_source, context="builder config post-build")
    for snapshot in activation_datasets:
        assert_file_snapshot(snapshot, context=f"activation dataset post-build {snapshot['dataset']}")
    for snapshot in checkpoint_files:
        assert_file_snapshot(
            snapshot,
            context=(
                f"checkpoint post-build {snapshot['dataset']}/{snapshot['lineage_id']}/"
                f"{snapshot['checkpoint_index_zero_based']}"
            ),
        )
    for snapshot in builder_source_snapshots:
        assert_file_snapshot(snapshot, context="operator-bank builder source post-build")
    validate_operator_bank_builder_source_seal(builder_source_seal)


def _builder_runtime_ledger(summary_device: torch.device) -> dict[str, Any]:
    """Bind interpreter/numeric-library versions that can affect emitted bytes."""

    return {
        "device": str(summary_device),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(summary_device)
            if summary_device.type == "cuda"
            else "cpu"
        ),
        "intraop_threads": int(torch.get_num_threads()),
        "numeric_policy": _numeric_runtime_policy(),
    }


@contextmanager
def _exact_gauge_verification_backend(device: torch.device) -> Iterator[dict[str, Any]]:
    """Disable approximate TF32 kernels only while checking gauge algebra.

    Channel gauges reorder convolution reductions.  CUDA TF32 kernels are not
    numerically equivariant to that reordering even when the transformed graph
    is algebraically exact, so they are inappropriate for this strict gate.
    The caller's backend policy is always restored, including on exceptions.
    """

    original_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    original_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    policy = {
        "device_type": device.type,
        "verification_cudnn_allow_tf32": False if device.type == "cuda" else original_cudnn_tf32,
        "verification_matmul_allow_tf32": False if device.type == "cuda" else original_matmul_tf32,
        "original_cudnn_allow_tf32": original_cudnn_tf32,
        "original_matmul_allow_tf32": original_matmul_tf32,
        "deterministic_algorithms_forced": False,
    }
    try:
        if device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        yield policy
    finally:
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32


@contextmanager
def _temporary_intraop_threads(threads: int) -> Iterator[dict[str, int]]:
    """Use a measured thread budget for the dedicated bank-build process.

    The context-summary workload consists of many small batched reductions.
    Letting PyTorch use the host-wide default (32 on the production machine)
    causes severe thread-pool oversubscription.  Keep the choice explicit and
    restore process state for callers that invoke the builder as a library.
    """

    requested = int(threads)
    if requested < 1:
        raise ValueError(f"context_summary_cpu_threads must be >=1, got {requested}")
    original = int(torch.get_num_threads())
    try:
        if requested != original:
            torch.set_num_threads(requested)
        yield {"original_intraop_threads": original, "effective_intraop_threads": requested}
    finally:
        if torch.get_num_threads() != original:
            torch.set_num_threads(original)


def _hash_record(arrays: dict[str, np.ndarray], metadata: dict[str, Any], prefix: str) -> str:
    digest = hashlib.sha256(json.dumps(redact_secrets(metadata), sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return f"{prefix}:{digest.hexdigest()}"


def _activation_dataset_inventory(
    rows: Iterable[dict[str, Any]], dataset_root: Path
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        path = (dataset_root / dataset / "dataset.pt").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"activation dataset is missing: {path}")
        inventory.append({"dataset": dataset, **sha256_file_stable(path)})
    return inventory


def _checkpoint_file_inventory(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = []
    for row in rows:
        path = Path(row["checkpoint_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {path}")
        actual = sha256_file(path)
        expected = str(row["checkpoint_sha256"])
        if actual != expected:
            raise RuntimeError(
                f"checkpoint hash mismatch: path={path} expected={expected} actual={actual}"
            )
        inventory.append(
            {
                "dataset": str(row["dataset"]),
                "lineage_id": str(row["lineage_id"]),
                "checkpoint_index_zero_based": int(row["checkpoint_index_zero_based"]),
                "path": str(path),
                "sha256": actual,
            }
        )
    return inventory


def _numeric_runtime_policy() -> dict[str, Any]:
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _validate_array_bank(
    root: Path,
    expected_manifest_sha256: str,
    expected_contract: dict[str, Any],
) -> None:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = validate_array_bank_contents(root, expected_manifest_sha256)
    if canonical_json_bytes(manifest.get("contract")) != canonical_json_bytes(expected_contract):
        raise RuntimeError(f"cached array-bank embedded contract mismatch: {manifest_path}")


def _load_validated_cache(
    cache_path: Path,
    *,
    contract_hash: str,
    build_contract: dict[str, Any],
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    cached = json.loads(cache_path.read_text())
    if cached.get("contract_sha256") != contract_hash:
        return None
    required = ("context_bank", "weight_tile_bank", "pair_manifest", "pair_manifest_sha256")
    missing = [key for key in required if key not in cached]
    if missing:
        raise RuntimeError(f"matching bank cache is incomplete: missing={missing} path={cache_path}")
    pair_path = Path(cached["pair_manifest"]).resolve()
    if not pair_path.is_file() or sha256_file(pair_path) != cached["pair_manifest_sha256"]:
        raise RuntimeError(f"matching pair manifest is missing or corrupt: {pair_path}")
    pair = json.loads(pair_path.read_text())
    if (
        pair.get("contract_sha256") != contract_hash
        or canonical_json_bytes(pair.get("contract")) != canonical_json_bytes(build_contract)
    ):
        raise RuntimeError(f"matching pair manifest has wrong embedded contract: {pair_path}")
    if Path(pair["context_bank"]).resolve() != Path(cached["context_bank"]).resolve():
        raise RuntimeError("cached context-bank reference disagrees with pair manifest")
    if Path(pair["weight_tile_bank"]).resolve() != Path(cached["weight_tile_bank"]).resolve():
        raise RuntimeError("cached weight-bank reference disagrees with pair manifest")
    _validate_array_bank(
        Path(pair["context_bank"]),
        str(pair["context_bank_manifest_sha256"]),
        {**build_contract, "kind": "context_bank"},
    )
    _validate_array_bank(
        Path(pair["weight_tile_bank"]),
        str(pair["weight_tile_bank_manifest_sha256"]),
        {
            **build_contract,
            "kind": "weight_tile_bank",
            "context_bank": str(Path(pair["context_bank"]).resolve()),
        },
    )
    for raw_path, digest in pair.get("permutation_files", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"cached permutation evidence is missing or corrupt: {path}")
    coverage_path = Path(pair["coverage_path"])
    if not coverage_path.is_file() or sha256_file(coverage_path) != pair.get("coverage_sha256"):
        raise RuntimeError(f"cached coverage evidence is missing or corrupt: {coverage_path}")
    return cached


def load_checkpoint_rows(
    path: Path,
    allowed_splits: tuple[str, ...],
    primary_checkpoint_indices: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tuple(allowed_splits) != ("train",):
        raise ValueError(f"production operator bank requires checkpoint_splits=('train',), got {allowed_splits}")
    if tuple(primary_checkpoint_indices) != (43, 44):
        raise ValueError(
            "production operator bank requires primary_checkpoint_indices=(43, 44), "
            f"got {primary_checkpoint_indices}"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("is_primary") and row.get("split") in allowed_splits]
    selected, inventory = validate_operator_checkpoint_selection(
        selected,
        expected_datasets=(task.key for task in SOURCE_TASKS),
        expected_lineages_per_dataset=35,
        expected_indices=tuple(primary_checkpoint_indices),
        expected_split="train",
    )
    selected.sort(key=lambda row: (row["dataset"], row["seed"], row["checkpoint_index_zero_based"]))
    inventory_by_key = {
        (row["dataset"], row["lineage_id"], row["checkpoint_index_zero_based"]): row
        for row in inventory
    }
    inventory = [
        inventory_by_key[
            (row["dataset"], row["lineage_id"], row["checkpoint_index_zero_based"])
        ]
        for row in selected
    ]
    return selected, inventory


def deterministic_train_batches(dataset: Any, *, images: int, batch_size: int, seed: int, device: str) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    count = min(int(images), len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count]
    for start in range(0, count, batch_size):
        idx = indices[start : start + batch_size]
        yield dataset.data[idx].to(device, non_blocking=True), dataset.targets[idx].to(device, non_blocking=True)


@torch.inference_mode()
def verify_gauge_views(
    model: ResNet18Slim,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    *,
    seed_parts: tuple[object, ...],
    views: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    model.load_state_dict(state, strict=True)
    model.eval()
    device = images.device
    original_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    original_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    runtime_reference = model(images)
    with _exact_gauge_verification_backend(device) as verification_policy:
        reference = model(images)
    result = []
    try:
        for view in range(views):
            gauge = sample_resnet_gauge(model, stable_seed(*seed_parts, "gauge", view))
            transformed = apply_resnet_gauge(state, gauge)
            inverse = invert_resnet_gauge(gauge)
            roundtrip = apply_resnet_gauge(transformed, inverse)
            roundtrip_bad = [key for key in state if not torch.equal(state[key], roundtrip[key])]
            if roundtrip_bad:
                raise ValueError(
                    f"graph gauge inverse is not bitwise exact: view={view} keys={roundtrip_bad[:5]}"
                )
            model.load_state_dict(transformed, strict=True)
            # Keep the deployment backend's finite-precision drift as telemetry,
            # but do not use approximate TF32 arithmetic as the algebraic gate.
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            runtime_candidate = model(images)
            with _exact_gauge_verification_backend(device):
                candidate = model(images)
            close = torch.allclose(reference, candidate, atol=atol, rtol=rtol)
            max_abs = float((reference - candidate).abs().max().item())
            runtime_max_abs = float((runtime_reference - runtime_candidate).abs().max().item())
            if not close:
                raise ValueError(
                    f"graph gauge view is not functionally equivalent under FP32 verification: "
                    f"view={view} max_abs={max_abs}"
                )
            result.append(
                {
                    "view_index": view,
                    "gauge_id": gauge_id(gauge),
                    "groups": {key: value.tolist() for key, value in gauge.items()},
                    "state_inverse_roundtrip_bitwise": True,
                    "functional_max_abs": max_abs,
                    "functional_atol": atol,
                    "functional_rtol": rtol,
                    "functional_verification_backend": verification_policy,
                    "runtime_backend_max_abs": runtime_max_abs,
                    "runtime_backend_argmax_equal": bool(
                        torch.equal(runtime_reference.argmax(dim=1), runtime_candidate.argmax(dim=1))
                    ),
                }
            )
    finally:
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        model.load_state_dict(state, strict=True)
    return result


def build_banks(config: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    threads = int(config["capture"].get("context_summary_cpu_threads", 4))
    bank_root = Path(config["paths"]["bank_root"])
    with _exclusive_operator_bank_build_lock(bank_root):
        try:
            with _temporary_intraop_threads(threads):
                return _build_banks_impl(config, verbose=verbose)
        except BaseException:
            for bank_name in ("context_bank", "weight_tile_bank"):
                shutil.rmtree(bank_root / f".{bank_name}.building", ignore_errors=True)
            raise


def _build_banks_impl(config: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    protocol_values = dict(config.get("protocol", {}))
    for name in ("checkpoint_splits", "primary_checkpoint_indices"):
        if name in protocol_values:
            protocol_values[name] = tuple(protocol_values[name])
    protocol = OperatorDatasetProtocol(**protocol_values)
    protocol.validate()
    paths = config["paths"]
    checkpoint_manifest = Path(paths["checkpoint_manifest"])
    dataset_root = Path(paths["dataset_pt_root"])
    bank_root = Path(paths["bank_root"])
    checkpoint_manifest_snapshot = sha256_file_stable(checkpoint_manifest)
    config_source_path = str(config.get("_builder_config_path", "") or "").strip()
    config_source_snapshot = (
        sha256_file_stable(Path(config_source_path)) if config_source_path else None
    )
    builder_source = operator_bank_builder_source_seal()
    validate_operator_bank_builder_source_seal(builder_source)
    builder_source_snapshots = [
        sha256_file_stable(
            WORKSPACE / str(item["path"]),
            expected_sha256=str(item["sha256"]),
        )
        for item in builder_source["files"]
    ]
    rows, checkpoint_files = load_checkpoint_rows(
        checkpoint_manifest,
        protocol.checkpoint_splits,
        protocol.primary_checkpoint_indices,
    )
    assert_file_snapshot(checkpoint_manifest_snapshot, context="checkpoint manifest")
    activation_datasets = _activation_dataset_inventory(rows, dataset_root)
    # All population/path/hash gates above run before the first output write.
    bank_root.mkdir(parents=True, exist_ok=True)
    summary_device = torch.device(config["capture"].get("context_summary_device", "cpu"))
    if summary_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured context summary device is unavailable: {summary_device}")
    summary_runtime = _builder_runtime_ledger(summary_device)
    checkpoint_hash = str(checkpoint_manifest_snapshot["sha256"])
    build_contract = {
        "checkpoint_manifest": str(checkpoint_manifest.resolve()),
        "checkpoint_manifest_sha256": checkpoint_hash,
        "checkpoint_manifest_stat": checkpoint_manifest_snapshot["stat"],
        "checkpoint_files": checkpoint_files,
        "activation_dataset_root": str(dataset_root.resolve()),
        "activation_datasets": activation_datasets,
        "builder_config_source": config_source_snapshot,
        "builder_config_payload_sha256": hashlib.sha256(
            canonical_json_bytes(
                {key: value for key, value in config.items() if key != "_builder_config_path"}
            )
        ).hexdigest(),
        "builder_source_implementation": builder_source,
        "protocol": protocol_values,
        "activation_images_per_checkpoint": int(config["capture"]["train_images_per_checkpoint"]),
        "activation_batch_size": int(config["capture"]["batch_size"]),
        "graph_permutation_views_are_metadata_only": True,
        "context_summary_algorithm": "batched_float32_v4",
        "context_summary_runtime": summary_runtime,
        "gauge_verifier_schema": "resnet_graph_gauge_fp32_v2_inverse_bitwise",
        "gauge_verifier": {
            "equivalence_images": int(config["permutations"]["equivalence_images"]),
            "atol": float(config["permutations"]["atol"]),
            "rtol": float(config["permutations"]["rtol"]),
            "strict_cuda_cudnn_allow_tf32": False,
            "strict_cuda_matmul_allow_tf32": False,
        },
    }
    contract_hash = hashlib.sha256(json.dumps(build_contract, sort_keys=True).encode()).hexdigest()
    cache_path = bank_root / "build_state.json"
    cached = _load_validated_cache(
        cache_path,
        contract_hash=contract_hash,
        build_contract=build_contract,
    )
    if cached is not None:
        print(f"[bank] verified cache hit contract={contract_hash}: {cache_path}")
        print(f"[bank] reused context={cached['context_bank']} weights={cached['weight_tile_bank']}")
        return cached

    print(f"[bank] selected checkpoints={len(rows)} split={protocol.checkpoint_splits} manifest_sha256={checkpoint_hash}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    context_writer = ImmutableArrayShardWriter(bank_root, "context_bank", protocol.context_shard_records)
    weight_writer = ImmutableArrayShardWriter(bank_root, "weight_tile_bank", protocol.weight_shard_records)
    permutation_records: list[dict[str, Any]] = []
    coverage_payload: dict[str, Any] | None = None
    context_count = weight_count = 0
    stage_seconds = {"activation_capture": 0.0, "gauge_verification": 0.0, "context_summary": 0.0, "record_serialization": 0.0}
    current_dataset = None
    dataset_payload = None
    checkpoint_snapshots = {
        (
            item["dataset"],
            item["lineage_id"],
            item["checkpoint_index_zero_based"],
        ): item
        for item in checkpoint_files
    }
    started = time.monotonic()
    for checkpoint_index, row in enumerate(tqdm(rows, desc="operator bank", disable=not verbose), 1):
        if row["dataset"] != current_dataset:
            current_dataset = row["dataset"]
            dataset_path = dataset_root / current_dataset / "dataset.pt"
            dataset_snapshot = next(
                item for item in activation_datasets if item["dataset"] == current_dataset
            )
            print(f"[bank:data] loading {dataset_path} sha256={dataset_snapshot['sha256']}")
            dataset_payload = _assert_snapshot_around_load(
                dataset_snapshot,
                lambda path: load_cached_dataset_payload(str(path)),
                context=f"activation dataset {current_dataset}",
            )
        assert dataset_payload is not None
        classes = int(torch.unique(torch.cat([part.targets for part in dataset_payload.values()])).numel())
        model = ResNet18Slim(3, classes, dropout=0.15, init_type=None, width_mult=0.5).to(device)
        checkpoint_snapshot = checkpoint_snapshots[
            (row["dataset"], row["lineage_id"], row["checkpoint_index_zero_based"])
        ]
        state = _assert_snapshot_around_load(
            checkpoint_snapshot,
            lambda path: torch.load(path, map_location=device, weights_only=True),
            context=(
                f"checkpoint {row['dataset']}/{row['lineage_id']}/"
                f"{row['checkpoint_index_zero_based']}"
            ),
        )
        model.load_state_dict(state, strict=True)
        specs = supported_operator_specs(model)
        coverage = assert_complete_coverage(model)
        if coverage_payload is None:
            coverage_payload = coverage
        elif coverage != coverage_payload:
            raise ValueError("state-key coverage changed between checkpoints")
        seed_parts = (row["dataset"], row["lineage_id"], row["checkpoint_index_zero_based"])
        batch_iter = deterministic_train_batches(
            dataset_payload["trainset"],
            images=int(config["capture"]["train_images_per_checkpoint"]),
            batch_size=int(config["capture"]["batch_size"]),
            seed=stable_seed(*seed_parts, "train-images"),
            device=device,
        )
        stage_started = time.monotonic()
        activations = capture_native_train_activations(
            model,
            batch_iter,
            specs,
            max_rows=protocol.activation_rows,
            seed_parts=seed_parts,
            device=device,
        )
        stage_seconds["activation_capture"] += time.monotonic() - stage_started
        fixed_count = min(int(config["permutations"]["equivalence_images"]), len(dataset_payload["trainset"]))
        fixed_images = dataset_payload["trainset"].data[:fixed_count].to(device)
        stage_started = time.monotonic()
        views = verify_gauge_views(
            model,
            state,
            fixed_images,
            seed_parts=seed_parts,
            views=protocol.permutation_views,
            atol=float(config["permutations"]["atol"]),
            rtol=float(config["permutations"]["rtol"]),
        )
        stage_seconds["gauge_verification"] += time.monotonic() - stage_started
        permutation_records.append({**{key: row[key] for key in ("dataset", "lineage_id", "seed", "split", "checkpoint_index_zero_based", "checkpoint_sha256")}, "views": views})

        stage_started = time.monotonic()
        contexts_for_spec = context_tiles_batched(
            {spec.key: activations[spec.key] for spec in specs},
            tile_width=protocol.tile_rows,
            max_rows=protocol.activation_rows,
            num_quantiles=protocol.quantiles,
            compute_device=summary_device,
        )
        stage_seconds["context_summary"] += time.monotonic() - stage_started

        stage_started = time.monotonic()
        for spec in specs:
            contexts_by_row: dict[int, str] = {}
            for context in contexts_for_spec[spec.key]:
                arrays = {
                    "raw_rows": context.raw_rows.numpy(),
                    "sample_mask": context.sample_mask.numpy(),
                    "feature_mask": context.feature_mask.numpy(),
                    "quantiles": context.quantiles.numpy(),
                    "mean": context.mean.numpy(),
                    "log_scale": context.log_scale.numpy(),
                    "covariance": context.covariance.numpy(),
                }
                metadata = {
                    "dataset": row["dataset"],
                    "lineage_id": row["lineage_id"],
                    "split": row["split"],
                    "checkpoint_index_zero_based": row["checkpoint_index_zero_based"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "layer_key": spec.key,
                    "input_row_start": context.row_start,
                    "valid_features": context.valid_features,
                    "native_activation_source": "dataset.trainset",
                }
                context_id = _hash_record(arrays, metadata, "context")
                metadata["context_id"] = context_id
                context_writer.add(arrays, metadata)
                contexts_by_row[context.row_start] = context_id
                context_count += 1
            matrix = parameter_to_matrix(state[spec.key], spec).to("cpu", torch.float32)
            for tile, mask, tile_spec in tile_matrix(matrix, protocol.tile_rows, protocol.tile_cols):
                context_id = contexts_by_row[tile_spec.row_start]
                arrays = {"weight": tile.numpy(), "mask": mask.cpu().numpy()}
                metadata = {
                    "dataset": row["dataset"],
                    "lineage_id": row["lineage_id"],
                    "split": row["split"],
                    "checkpoint_index_zero_based": row["checkpoint_index_zero_based"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "layer_key": spec.key,
                    "operator": spec.to_dict(),
                    "tile": {
                        "row_start": tile_spec.row_start,
                        "col_start": tile_spec.col_start,
                        "valid_rows": tile_spec.valid_rows,
                        "valid_cols": tile_spec.valid_cols,
                    },
                    "context_id": context_id,
                    "lineage_sampling_weight": 0.5,
                }
                metadata["weight_tile_id"] = _hash_record(arrays, metadata, "weight-tile")
                weight_writer.add(arrays, metadata)
                weight_count += 1
        stage_seconds["record_serialization"] += time.monotonic() - stage_started
        if verbose:
            elapsed = time.monotonic() - started
            print(
                f"[bank] checkpoint={checkpoint_index}/{len(rows)} dataset={row['dataset']} "
                f"lineage={row['lineage_id']} contexts={context_count} weight_tiles={weight_count} "
                f"elapsed={elapsed:.1f}s rate={checkpoint_index / max(elapsed, 1e-9):.3f} checkpoint/s "
                f"stage_seconds={json.dumps(stage_seconds, sort_keys=True)}",
                flush=True,
            )

    # One final no-I/O-write barrier detects every input/source/config identity
    # drift before either array-bank writer can publish its immutable directory.
    _assert_build_input_snapshots(
        checkpoint_manifest=checkpoint_manifest_snapshot,
        config_source=config_source_snapshot,
        activation_datasets=activation_datasets,
        checkpoint_files=checkpoint_files,
        builder_source_snapshots=builder_source_snapshots,
        builder_source_seal=builder_source,
    )

    context_root = context_writer.finalize({**build_contract, "kind": "context_bank"})
    weight_root = weight_writer.finalize(
        {
            **build_contract,
            "kind": "weight_tile_bank",
            "context_bank": str(context_root.resolve()),
        }
    )
    permutation_files = write_records_immutable(bank_root / f"permutation_views-{contract_hash[:16]}", permutation_records)
    coverage_path = bank_root / f"coverage-{contract_hash[:16]}.json"
    write_json_immutable(coverage_path, coverage_payload)
    pair_manifest_path = bank_root / f"operator_dataset-{contract_hash[:16]}.json"
    pair_manifest = {
        "format_version": 1,
        "contract": build_contract,
        "contract_sha256": contract_hash,
        "context_bank": str(context_root.resolve()),
        "context_bank_manifest_sha256": sha256_file(context_root / "manifest.json"),
        "weight_tile_bank": str(weight_root.resolve()),
        "weight_tile_bank_manifest_sha256": sha256_file(weight_root / "manifest.json"),
        "context_records": context_count,
        "weight_tile_records": weight_count,
        "permutation_files": permutation_files,
        "coverage_path": str(coverage_path.resolve()),
        "coverage_sha256": sha256_file(coverage_path),
        "elapsed_seconds": time.monotonic() - started,
        "stage_seconds": stage_seconds,
        "activation_tensors_are_not_duplicated_per_output_tile": True,
    }
    write_json_immutable(pair_manifest_path, pair_manifest)
    cache = {
        "contract_sha256": contract_hash,
        "context_bank": str(context_root.resolve()),
        "weight_tile_bank": str(weight_root.resolve()),
        "pair_manifest": str(pair_manifest_path.resolve()),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
    }
    write_json_atomic(cache_path, cache)
    print(f"[bank:done] contexts={context_count} weights={weight_count}")
    print(f"[bank:done] context={context_root} weights={weight_root} manifest={pair_manifest_path}")
    return cache


class MMapOperatorDataset(Dataset):
    """Reference mmap reader proving the bank has no per-tile context copies."""

    def __init__(self, pair_manifest: str | Path, hot_shards: int = 4) -> None:
        pair = json.loads(Path(pair_manifest).read_text())
        self.context_root = Path(pair["context_bank"])
        self.weight_root = Path(pair["weight_tile_bank"])
        self.hot_shards = int(hot_shards)
        self._arrays: OrderedDict[tuple[str, int], dict[str, np.ndarray]] = OrderedDict()
        self.context_locations: dict[str, tuple[int, int]] = {}
        self.weight_rows: list[tuple[int, int, dict[str, Any]]] = []
        self._index_banks()

    @staticmethod
    def _shards(root: Path) -> list[Path]:
        return sorted(path for path in root.glob("shard-*") if path.is_dir())

    def _index_banks(self) -> None:
        for shard_index, shard in enumerate(self._shards(self.context_root)):
            for offset, line in enumerate((shard / "records.jsonl").read_text().splitlines()):
                metadata = json.loads(line)
                self.context_locations[metadata["context_id"]] = (shard_index, offset)
        for shard_index, shard in enumerate(self._shards(self.weight_root)):
            for offset, line in enumerate((shard / "records.jsonl").read_text().splitlines()):
                self.weight_rows.append((shard_index, offset, json.loads(line)))

    def __len__(self) -> int:
        return len(self.weight_rows)

    def _load(self, kind: str, shard_index: int) -> dict[str, np.ndarray]:
        key = (kind, shard_index)
        if key in self._arrays:
            self._arrays.move_to_end(key)
            return self._arrays[key]
        root = self.context_root if kind == "context" else self.weight_root
        shard = self._shards(root)[shard_index]
        arrays = {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in shard.glob("*.npy")}
        self._arrays[key] = arrays
        while len(self._arrays) > self.hot_shards:
            self._arrays.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        weight_shard, weight_offset, metadata = self.weight_rows[index]
        context_shard, context_offset = self.context_locations[metadata["context_id"]]
        weight_arrays = self._load("weight", weight_shard)
        context_arrays = self._load("context", context_shard)
        return {
            "weight": torch.from_numpy(np.array(weight_arrays["weight"][weight_offset], copy=True)),
            "weight_mask": torch.from_numpy(np.array(weight_arrays["mask"][weight_offset], copy=True)),
            "context": torch.from_numpy(np.array(context_arrays["raw_rows"][context_offset], copy=True)),
            "sample_mask": torch.from_numpy(np.array(context_arrays["sample_mask"][context_offset], copy=True)),
            "metadata": metadata,
        }


def collate_operator_rows(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.stack([row["weight"] for row in rows]),
        "context": torch.stack([row["context"] for row in rows]),
    }


def profile_bank(
    config: dict[str, Any],
    *,
    steps: int,
    verbose: bool,
    warmup_steps: int | None = None,
) -> dict[str, Any]:
    cache = json.loads((Path(config["paths"]["bank_root"]) / "build_state.json").read_text())
    dataset = MMapOperatorDataset(cache["pair_manifest"], hot_shards=int(config["loader"]["hot_shards"]))
    measured_steps = max(1, int(steps))
    warmup = max(
        0,
        int(config["loader"].get("profile_warmup_steps", 50) if warmup_steps is None else warmup_steps),
    )
    batch_size = int(config["loader"]["batch_size"])
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=(warmup + measured_steps) * batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(config["loader"]["workers"]),
        persistent_workers=int(config["loader"]["workers"]) > 0,
        pin_memory=True,
        prefetch_factor=int(config["loader"]["prefetch_factor"]) if int(config["loader"]["workers"]) > 0 else None,
        collate_fn=collate_operator_rows,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    input_wait = h2d = gpu = 0.0
    start = previous = time.perf_counter()
    completed = 0
    iterator = iter(loader)
    for raw_step in range(warmup + measured_steps):
        batch = next(iterator)
        loaded = time.perf_counter()
        input_wait += loaded - previous
        transfer_start = loaded
        weight = batch["weight"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        transferred = time.perf_counter()
        h2d += transferred - transfer_start
        # Production-shaped memory traffic plus a nontrivial reduction.  This
        # is an I/O gate, not an AE throughput claim.
        value = (weight.square().mean() + context.square().mean())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        finished = time.perf_counter()
        if raw_step < warmup:
            if raw_step + 1 == warmup:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                input_wait = h2d = gpu = 0.0
                start = finished = time.perf_counter()
            previous = finished
            continue
        gpu += finished - transferred
        previous = finished
        completed += 1
        if verbose and completed % 50 == 0:
            print(
                f"[profile] measured_step={completed}/{measured_steps} warmup={warmup} "
                f"value={float(value):.5g} input_wait={input_wait:.3f}s "
                f"h2d={h2d:.3f}s gpu={gpu:.3f}s"
            )
    wall = time.perf_counter() - start
    report = {
        "steps": completed,
        "warmup_steps": warmup,
        "batch_size": batch_size,
        "wall_seconds": wall,
        "steps_per_second": completed / wall,
        "samples_per_second": completed * batch_size / wall,
        "input_wait_ms_per_batch": 1000.0 * input_wait / completed,
        "h2d_ms_per_batch": 1000.0 * h2d / completed,
        "input_wait_fraction": input_wait / wall,
        "h2d_fraction": h2d / wall,
        "gpu_active_fraction": gpu / wall,
        "reader_only_input_wait_le_0.05_of_trivial_loop": input_wait / wall <= 0.05,
        "iterator_restarts_during_measurement": 0,
        # The true >=90% active gate must be measured with the actual AE step;
        # this reader-only profiler cannot establish it.
        "ae_gpu_active_gate_established": False,
    }
    output = Path(config["paths"]["bank_root"]) / "throughput_profile.json"
    write_json_atomic(output, report)
    print(f"[profile:done] {json.dumps(report, sort_keys=True)} -> {output}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "profile"))
    parser.add_argument("--config", type=Path, default=WORKSPACE / "conf/weightclip_benchmark/operator_dataset.yaml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_snapshot = sha256_file_stable(args.config)
    config = yaml.safe_load(args.config.read_text())
    assert_file_snapshot(config_snapshot, context="operator-bank builder config load")
    config["_builder_config_path"] = str(args.config.resolve())
    verbose = not args.quiet
    print(f"[startup] stage={args.stage} config={json.dumps(redact_secrets(config), sort_keys=True)}")
    print(f"[startup] device={'cuda:0' if torch.cuda.is_available() else 'cpu'} dtype=float32 cache_mode=content-addressed")
    if args.stage == "build":
        build_banks(config, verbose=verbose)
    else:
        profile_bank(config, steps=args.steps, verbose=verbose, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
