#!/usr/bin/env python3
"""Materialize and train the frozen WeightCLIP ResNet18Slim zoo contract.

Stages are explicit and resumable: ``materialize``, ``materialize-ood``,
``sweep``, ``train``, ``manifest``, or ``all``.  ``all`` does not launch any
Weight-AE training and intentionally does not unseal OOD evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import queue as std_queue
import sys
import tarfile
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from big_vae.weightclip_benchmark.manifests import (  # noqa: E402
    CheckpointRecord,
    LineageRecord,
    sha256_file,
    stable_lineage_split,
    validate_lineage_records,
    write_json_atomic,
    write_json_immutable,
    write_records_immutable,
)
from big_vae.weightclip_benchmark.dataset_payload import (  # noqa: E402
    CachedDataset,
    load_cached_dataset_payload,
)
from big_vae.weightclip_benchmark.metadata import (  # noqa: E402
    OOD_ARCHIVE_TASKS,
    OOD_CLASS_COUNTS,
    SOURCE_TASKS,
    ZooProtocol,
    frozen_contract,
    redact_secrets,
)
from big_vae.weightclip_benchmark.resnet18slim import ResNet18Slim, count_parameters  # noqa: E402


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_dataset(path: Path) -> dict[str, CachedDataset]:
    return load_cached_dataset_payload(str(path))


def _tensor_from_image(data: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB").resize((32, 32), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def materialize_archive(archive: Path, dataset_root: Path, *, overwrite: bool, verbose: bool) -> dict[str, Any]:
    """Decode all ten paper tasks from the official TANS archive in one pass."""

    raw_to_task = {raw: task for task in SOURCE_TASKS for raw in task.raw_tasks}
    existing = [task for task in SOURCE_TASKS if (dataset_root / task.key / "dataset.pt").exists()]
    if len(existing) == len(SOURCE_TASKS) and not overwrite:
        print(f"[materialize] cache hit: all ten dataset.pt files under {dataset_root}")
        return json.loads((dataset_root / "manifest.json").read_text())

    buffers: dict[str, dict[str, list[tuple[str, torch.Tensor, str]]]] = {
        task.key: {"tr": [], "va": [], "te": []} for task in SOURCE_TASKS
    }
    selected_files: list[str] = []
    archive_sha = sha256_file(archive)
    print(f"[materialize] archive={archive} sha256={archive_sha} size={archive.stat().st_size}")
    start = time.monotonic()
    with tarfile.open(archive, "r:gz") as handle:
        for member in tqdm(handle, desc="scan/decode official TANS archive", disable=not verbose):
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 8:
                continue
            raw_name, split_name, class_name = parts[4], parts[5], parts[6]
            task = raw_to_task.get(raw_name)
            if task is None or split_name not in ("tr", "va", "te"):
                continue
            stream = handle.extractfile(member)
            if stream is None:
                continue
            try:
                tensor = _tensor_from_image(stream.read())
            except Exception as exc:
                raise RuntimeError(f"failed decoding {member.name}") from exc
            buffers[task.key][split_name].append((class_name, tensor, member.name))
            selected_files.append(member.name)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "archive": str(archive.resolve()),
        "archive_sha256": archive_sha,
        "selected_member_list_sha256": __import__("hashlib").sha256("\n".join(sorted(selected_files)).encode()).hexdigest(),
        "preprocessing": "PIL RGB -> bilinear 32x32 -> float32 [0,1] -> [-1,1]",
        "datasets": {},
        "elapsed_seconds": time.monotonic() - start,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    for task in SOURCE_TASKS:
        class_names = sorted({class_name for split in buffers[task.key].values() for class_name, _, _ in split})
        class_to_index = {name: idx for idx, name in enumerate(class_names)}
        payload: dict[str, CachedDataset] = {}
        source_hash = __import__("hashlib").sha256()
        for raw_split, output_key in (("tr", "trainset"), ("va", "valset"), ("te", "testset")):
            rows = buffers[task.key][raw_split]
            if not rows:
                raise ValueError(f"paper task {task.key} has no {raw_split} images in {archive}")
            for _, _, member_name in sorted(rows, key=lambda row: row[2]):
                source_hash.update(member_name.encode() + b"\n")
            data = torch.stack([tensor for _, tensor, _ in rows]).contiguous()
            targets = torch.tensor([class_to_index[class_name] for class_name, _, _ in rows], dtype=torch.long)
            payload[output_key] = CachedDataset(data, targets)
        output = dataset_root / task.key / "dataset.pt"
        if output.exists() and not overwrite:
            print(f"[materialize:{task.key}] cache hit {output}")
        else:
            atomic_torch_save(payload, output)
        manifest["datasets"][task.key] = {
            "paper_name": task.paper_name,
            "raw_tasks": list(task.raw_tasks),
            "classes": len(class_names),
            "split_sizes": {key: len(value) for key, value in payload.items()},
            "source_file_list_sha256": source_hash.hexdigest(),
            "dataset_pt": str(output.resolve()),
            "dataset_pt_sha256": sha256_file(output),
        }
        print(f"[materialize:{task.key}] classes={len(class_names)} sizes={manifest['datasets'][task.key]['split_sizes']} -> {output}")
    write_json_atomic(dataset_root / "manifest.json", manifest)
    return manifest


def _member_list_sha256(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _payload_equal(path: Path, payload: dict[str, CachedDataset]) -> bool:
    """Return whether an existing dataset.pt is tensor-identical to payload."""

    try:
        existing = load_dataset(path)
    except Exception:
        return False
    return all(
        torch.equal(existing[key].data, payload[key].data)
        and torch.equal(existing[key].targets, payload[key].targets)
        for key in ("trainset", "valset", "testset")
    )


def _write_dataset_payload(
    payload: dict[str, CachedDataset],
    output: Path,
    *,
    overwrite: bool,
    label: str,
) -> None:
    if output.exists() and not overwrite:
        if not _payload_equal(output, payload):
            raise FileExistsError(
                f"{output} exists but is not tensor-identical to the verified {label} payload; "
                "pass --overwrite-datasets explicitly"
            )
        print(f"[materialize-ood:{label}] exact tensor cache hit {output}")
        return
    atomic_torch_save(payload, output)


def materialize_ood_archive(
    archive: Path,
    dataset_root: Path,
    *,
    expected_sha256: str,
    overwrite: bool,
    verbose: bool,
    actual_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Stream the exact five released OOD tasks without extracting the tarball."""

    actual_sha256 = sha256_file(archive) if actual_sha256 is None else actual_sha256
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"OOD archive SHA256 mismatch: expected={expected_sha256} actual={actual_sha256} archive={archive}"
        )
    raw_to_task = {task.raw_task: task for task in OOD_ARCHIVE_TASKS}
    buffers: dict[str, dict[str, list[tuple[str, torch.Tensor, str]]]] = {
        task.key: {"tr": [], "va": [], "te": []} for task in OOD_ARCHIVE_TASKS
    }
    print(
        f"[materialize-ood] verified archive={archive} sha256={actual_sha256} "
        f"size={archive.stat().st_size} mode=stream-no-extraction"
    )
    with tarfile.open(archive, "r:gz") as handle:
        for member in tqdm(handle, desc="scan/decode five official OOD tasks", disable=not verbose):
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 8:
                continue
            raw_name, split_name, class_name = parts[4], parts[5], parts[6]
            task = raw_to_task.get(raw_name)
            if task is None or split_name not in ("tr", "va", "te"):
                continue
            stream = handle.extractfile(member)
            if stream is None:
                continue
            try:
                tensor = _tensor_from_image(stream.read())
            except Exception as exc:
                raise RuntimeError(f"failed decoding {member.name}") from exc
            buffers[task.key][split_name].append((class_name, tensor, member.name))

    records: dict[str, dict[str, Any]] = {}
    dataset_root.mkdir(parents=True, exist_ok=True)
    for task in OOD_ARCHIVE_TASKS:
        train_classes = sorted({class_name for class_name, _, _ in buffers[task.key]["tr"]})
        if len(train_classes) != task.classes:
            raise ValueError(
                f"{task.key}: expected {task.classes} train classes, found {len(train_classes)}: {train_classes}"
            )
        class_to_index = {name: idx for idx, name in enumerate(train_classes)}
        payload: dict[str, CachedDataset] = {}
        selected_members: list[str] = []
        for raw_split, output_key in (("tr", "trainset"), ("va", "valset"), ("te", "testset")):
            rows = sorted(buffers[task.key][raw_split], key=lambda row: row[2])
            if not rows:
                raise ValueError(f"OOD task {task.key} has no {raw_split} images in {archive}")
            unknown_classes = sorted({class_name for class_name, _, _ in rows} - set(class_to_index))
            if unknown_classes:
                raise ValueError(f"{task.key}/{raw_split}: classes absent from train split: {unknown_classes}")
            payload[output_key] = CachedDataset(
                torch.stack([tensor for _, tensor, _ in rows]).contiguous(),
                torch.tensor([class_to_index[class_name] for class_name, _, _ in rows], dtype=torch.long),
            )
            selected_members.extend(member_name for _, _, member_name in rows)
        output = dataset_root / task.key / "dataset.pt"
        _write_dataset_payload(payload, output, overwrite=overwrite, label=task.key)
        records[task.key] = {
            "official_slug": task.key,
            "raw_task": task.raw_task,
            "classes": task.classes,
            "class_names": train_classes,
            "split_sizes": {key: len(value) for key, value in payload.items()},
            "source_member_list_sha256": _member_list_sha256(selected_members),
            "source_members": len(selected_members),
            "dataset_pt": str(output.resolve()),
            "dataset_pt_sha256": sha256_file(output),
            "provenance": "official TANS raw_m_test archive; streamed without extraction",
        }
        print(
            f"[materialize-ood:{task.key}] classes={task.classes} "
            f"sizes={records[task.key]['split_sizes']} -> {output}"
        )
    return records


def _cifar_tensor(data: Any) -> torch.Tensor:
    array = np.asarray(data, dtype=np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"unexpected torchvision CIFAR10 data shape: {array.shape}")
    # Exact value transform of Resize((32,32)) -> ToTensor -> Normalize(.5,.5).
    return torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5).div_(0.5)


def _cifar_split_indices(size: int, validation_fraction: float, split_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(split_seed))
    validation_size = int(size * validation_fraction)
    return permutation[validation_size:], permutation[:validation_size]


def _load_torchvision_cifar10(download_root: Path, *, download: bool) -> tuple[Any, Any, str]:
    try:
        import torchvision
        from torchvision.datasets import CIFAR10
    except ImportError as exc:  # pragma: no cover - depends on production environment
        raise RuntimeError(
            "materialize-ood requires torchvision for the official CIFAR10 path; install a torch-compatible torchvision"
        ) from exc
    train = CIFAR10(str(download_root), train=True, download=download)
    test = CIFAR10(str(download_root), train=False, download=download)
    return train, test, str(torchvision.__version__)


def _file_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return rows, hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def materialize_cifar10(
    dataset_root: Path,
    download_root: Path,
    *,
    validation_fraction: float,
    split_seed: int,
    download: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Use torchvision's official CIFAR10 source and reproduce its released split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("CIFAR10 validation_fraction must be in (0, 1)")
    train, test, torchvision_version = _load_torchvision_cifar10(download_root, download=download)
    train_data, train_targets = _cifar_tensor(train.data), torch.as_tensor(train.targets, dtype=torch.long)
    test_data, test_targets = _cifar_tensor(test.data), torch.as_tensor(test.targets, dtype=torch.long)
    if len(train_targets) != 50_000 or len(test_targets) != 10_000:
        raise ValueError(
            f"official CIFAR10 size mismatch: train={len(train_targets)} test={len(test_targets)}"
        )
    train_indices, validation_indices = _cifar_split_indices(
        len(train_targets), validation_fraction, split_seed
    )
    payload = {
        "trainset": CachedDataset(train_data[train_indices].contiguous(), train_targets[train_indices].contiguous()),
        "valset": CachedDataset(
            train_data[validation_indices].contiguous(), train_targets[validation_indices].contiguous()
        ),
        "testset": CachedDataset(test_data.contiguous(), test_targets.contiguous()),
    }
    output = dataset_root / "cifar10" / "dataset.pt"
    _write_dataset_payload(payload, output, overwrite=overwrite, label="cifar10")
    inventory, inventory_sha256 = _file_inventory(download_root)
    class_names = list(getattr(train, "classes", []))
    if len(class_names) != OOD_CLASS_COUNTS["cifar10"]:
        raise ValueError(f"official CIFAR10 class metadata mismatch: {class_names}")
    record = {
        "official_slug": "cifar10",
        "raw_task": "torchvision.datasets.CIFAR10",
        "classes": 10,
        "class_names": class_names,
        "split_sizes": {key: len(value) for key, value in payload.items()},
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "train_indices_sha256": hashlib.sha256(train_indices.numpy().tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(validation_indices.numpy().tobytes()).hexdigest(),
        "torchvision_version": torchvision_version,
        "download_root": str(download_root.resolve()),
        "raw_file_inventory": inventory,
        "raw_file_inventory_sha256": inventory_sha256,
        "dataset_pt": str(output.resolve()),
        "dataset_pt_sha256": sha256_file(output),
        "provenance": "torchvision.datasets.CIFAR10 official download path",
    }
    print(f"[materialize-ood:cifar10] sizes={record['split_sizes']} -> {output}")
    return record


def materialize_ood_datasets(
    archive: Path,
    dataset_root: Path,
    download_root: Path,
    *,
    expected_sha256: str,
    validation_fraction: float,
    split_seed: int,
    download: bool,
    overwrite: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Materialize all six sealed OOD datasets; this does not evaluate them."""

    actual_archive_sha256 = sha256_file(archive)
    if actual_archive_sha256 != expected_sha256:
        raise ValueError(
            f"OOD archive SHA256 mismatch: expected={expected_sha256} "
            f"actual={actual_archive_sha256} archive={archive}"
        )
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            cached = json.loads(manifest_path.read_text())
            records = cached["datasets"]
            valid = (
                cached.get("stage") == "materialize-ood"
                and cached.get("evaluation_status") == "SEALED_NOT_EVALUATED"
                and cached.get("archive_sha256") == actual_archive_sha256
                and cached.get("official_slugs") == list(OOD_CLASS_COUNTS)
                and set(records) == set(OOD_CLASS_COUNTS)
                and all(
                    Path(record["dataset_pt"]).is_file()
                    and sha256_file(Path(record["dataset_pt"])) == record["dataset_pt_sha256"]
                    for record in records.values()
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if valid:
            print(f"[materialize-ood] exact sealed cache hit {manifest_path}")
            return cached
        print(f"[materialize-ood] stale/incomplete cache ignored: {manifest_path}")

    started = time.monotonic()
    datasets = materialize_ood_archive(
        archive,
        dataset_root,
        expected_sha256=expected_sha256,
        overwrite=overwrite,
        verbose=verbose,
        actual_sha256=actual_archive_sha256,
    )
    datasets["cifar10"] = materialize_cifar10(
        dataset_root,
        download_root,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        download=download,
        overwrite=overwrite,
    )
    manifest = {
        "format_version": 1,
        "stage": "materialize-ood",
        "evaluation_status": "SEALED_NOT_EVALUATED",
        "archive": str(archive.resolve()),
        "archive_sha256": actual_archive_sha256,
        "expected_archive_sha256": expected_sha256,
        "preprocessing": "RGB/official CIFAR -> 32x32 float32 -> Normalize(mean=.5,std=.5) [-1,1]",
        "official_slugs": list(OOD_CLASS_COUNTS),
        "class_counts": OOD_CLASS_COUNTS,
        "conditioning_contract": {
            "allowed_inputs": ["trainset_images", "dataset_embedding", "train_only_context"],
            "target_checkpoint_required": False,
            "target_checkpoint_forbidden": True,
            "conditioning_consumes_test_images": False,
        },
        "datasets": datasets,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"[materialize-ood] wrote sealed data manifest -> {manifest_path}")
    return manifest


def configure_runtime() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def evaluate(model: nn.Module, dataset: CachedDataset, batch_size: int, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, count = 0.0, 0, 0
    for images, targets in dataset.batches(batch_size, shuffle=False, device=device):
        logits = model(images)
        loss += float(criterion(logits, targets).item())
        correct += int((logits.argmax(1) == targets).sum().item())
        count += len(targets)
    return loss / count, 100.0 * correct / count


def train_epoch(model: nn.Module, dataset: CachedDataset, optimizer: SGD, scheduler: OneCycleLR, batch_size: int, device: str) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, count, batches = 0.0, 0, 0, 0
    for images, targets in dataset.batches(batch_size, shuffle=True, device=device):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.item())
        correct += int((logits.argmax(1) == targets).sum().item())
        count += len(targets)
        batches += 1
    return loss_sum / batches, 100.0 * correct / count


def _cpu_fp32_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().to("cpu", torch.float32).contiguous() for key, value in model.state_dict().items()}


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def finalize_lineage(lineage_dir: Path, summary: Mapping[str, Any]) -> None:
    """Commit completion before deleting the now-redundant rolling resume."""

    complete = lineage_dir / "complete.json"
    write_json_atomic(complete, dict(summary))
    # Re-read the atomic commit before releasing recoverability state. All 46
    # FP32 archive checkpoints remain; only optimizer/scheduler resume is gone.
    if json.loads(complete.read_text()) != dict(summary):
        raise RuntimeError(f"lineage completion commit verification failed: {complete}")
    (lineage_dir / "resume.pt").unlink(missing_ok=True)


def run_lineage(task: dict[str, Any], common: dict[str, Any], device: str) -> dict[str, Any]:
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset_path = Path(common["dataset_pt"])
    datasets = load_dataset(dataset_path)
    classes = int(torch.unique(torch.cat([value.targets for value in datasets.values()])).numel())
    model = ResNet18Slim(3, classes, "relu", common["dropout"], common["init_type"], common["width_mult"]).to(device)
    optimizer = SGD(model.parameters(), lr=task["lr"], momentum=common["momentum"], weight_decay=common["weight_decay"])
    steps_per_epoch = math.ceil(len(datasets["trainset"]) / common["batch_size"])
    scheduler = OneCycleLR(optimizer, max_lr=task["lr"], epochs=common["scheduler_epochs"], steps_per_epoch=steps_per_epoch)
    save = bool(task["save"])
    lineage_dir = Path(common["output_root"]) / common["dataset"] / f"lineage-{seed:04d}"
    progress_path = lineage_dir / "metrics.jsonl"
    resume_path = lineage_dir / "resume.pt"
    start_epoch = 0
    if save:
        lineage_dir.mkdir(parents=True, exist_ok=True)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            expected_resume = {
                "seed": seed,
                "lr": float(task["lr"]),
                "dataset_sha256": common["dataset_sha256"],
                "resolved_contract_sha256": hashlib.sha256(
                    json.dumps({**common, **task}, sort_keys=True).encode()
                ).hexdigest(),
            }
            if any(state.get(key) != value for key, value in expected_resume.items()):
                raise ValueError(f"resume contract mismatch: {resume_path}")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if "rng_state" not in state:
                raise ValueError(f"legacy non-exact resume is forbidden: {resume_path}")
            _restore_rng_state(state["rng_state"])
            start_epoch = int(state["next_epoch"])
            if progress_path.exists():
                # A crash can occur after appending metrics but before replacing
                # the rolling resume state.  Keep exactly the committed prefix.
                committed = [
                    line for line in progress_path.read_text().splitlines()
                    if int(json.loads(line)["epoch_one_based"]) <= start_epoch
                ]
                progress_path.write_text("\n".join(committed) + ("\n" if committed else ""))
            print(f"[{common['dataset']} seed={seed}] resume cache hit {resume_path} next_epoch={start_epoch}")
        else:
            partial_epochs = sorted((lineage_dir / "epochs").glob("epoch-*.pt"))
            if partial_epochs or (progress_path.exists() and progress_path.stat().st_size):
                raise RuntimeError(
                    f"partial lineage has no exact resume state: {lineage_dir}; "
                    "audit/remove only this incomplete lineage before retry"
                )
            atomic_torch_save(_cpu_fp32_state(model), lineage_dir / "initialization.pt")
            write_json_atomic(lineage_dir / "resolved_config.json", redact_secrets({**common, **task, "device": device, "dtype": "float32"}))
    started = time.monotonic()
    last: dict[str, Any] = {}
    if start_epoch and progress_path.exists():
        committed_rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
        if committed_rows:
            last = committed_rows[-1]
    for epoch_index in range(start_epoch, int(common["epochs"])):
        epoch_started = time.monotonic()
        train_loss, train_acc = train_epoch(model, datasets["trainset"], optimizer, scheduler, common["batch_size"], device)
        val_loss, val_acc = evaluate(model, datasets["valset"], common["batch_size"] * 2, device)
        test_loss, test_acc = evaluate(model, datasets["testset"], common["batch_size"] * 2, device)
        last = {
            "epoch_index_zero_based": epoch_index,
            "epoch_one_based": epoch_index + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        if save:
            atomic_torch_save(_cpu_fp32_state(model), lineage_dir / "epochs" / f"epoch-{epoch_index + 1:03d}.pt")
            with progress_path.open("a") as handle:
                handle.write(json.dumps(last, sort_keys=True) + "\n")
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "next_epoch": epoch_index + 1,
                    "seed": seed,
                    "lr": float(task["lr"]),
                    "dataset_sha256": common["dataset_sha256"],
                    "resolved_contract_sha256": hashlib.sha256(
                        json.dumps({**common, **task}, sort_keys=True).encode()
                    ).hexdigest(),
                    "rng_state": _rng_state(),
                },
                resume_path,
            )
        if common["verbose"]:
            print(
                f"[{common['dataset']} seed={seed} lr={task['lr']:.4g}] "
                f"epoch={epoch_index + 1}/{common['epochs']} train_loss={train_loss:.5f} "
                f"val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
                f"lr_now={last['learning_rate']:.6g} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    summary = {"dataset": common["dataset"], "seed": seed, "lr": float(task["lr"]), **last, "elapsed_seconds": time.monotonic() - started}
    if save:
        finalize_lineage(lineage_dir, summary)
    return summary


def worker(worker_id: int, device: str, tasks: list[dict[str, Any]], common: dict[str, Any], queue: mp.Queue) -> None:
    configure_runtime()
    torch.cuda.set_device(int(device.split(":")[-1]))
    for task in tasks:
        try:
            queue.put({"ok": True, "worker": worker_id, "result": run_lineage(task, common, device)})
        except Exception as exc:  # pragma: no cover - exercised in production
            queue.put({"ok": False, "worker": worker_id, "task": task, "error": repr(exc)})


def parallel_runs(tasks: list[dict[str, Any]], common: dict[str, Any], workers: int, description: str) -> list[dict[str, Any]]:
    if not tasks:
        return []
    count = min(workers, len(tasks))
    buckets = [tasks[index::count] for index in range(count)]
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes = [context.Process(target=worker, args=(index, "cuda:0", bucket, common, queue)) for index, bucket in enumerate(buckets)]
    for process in processes:
        process.start()
    results = []
    with tqdm(total=len(tasks), desc=description, disable=not common["verbose"]) as progress:
        while len(results) < len(tasks):
            try:
                result = queue.get(timeout=5.0)
            except std_queue.Empty:
                crashed = [process for process in processes if process.exitcode not in (None, 0)]
                if crashed:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    raise RuntimeError(
                        f"zoo worker died before reporting: pid={crashed[0].pid} exitcode={crashed[0].exitcode}"
                    )
                continue
            results.append(result)
            progress.set_postfix(status="ok" if result["ok"] else "FAILED")
            progress.update()
    for process in processes:
        process.join()
    failures = [row for row in results if not row["ok"]]
    if failures:
        raise RuntimeError(f"{len(failures)} lineage tasks failed; first={failures[0]}")
    return [row["result"] for row in results]


def run_sweep(dataset: str, dataset_pt: Path, output_root: Path, protocol: ZooProtocol, verbose: bool) -> float:
    sweep_path = output_root / dataset / "lr_sweep.json"
    if sweep_path.exists():
        payload = json.loads(sweep_path.read_text())
        expected = {
            "dataset_sha256": sha256_file(dataset_pt),
            "protocol_sha256": hashlib.sha256(json.dumps(protocol.to_dict(), sort_keys=True).encode()).hexdigest(),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"stale/unbound LR sweep cache: {sweep_path}; expected={expected}")
        print(f"[sweep:{dataset}] cache hit {sweep_path}; selected={payload['chosen_lr']}")
        return float(payload["chosen_lr"])
    common = _common(dataset, dataset_pt, output_root, protocol, verbose)
    tasks = [
        {"seed": seed, "lr": lr, "save": False}
        for lr in protocol.lr_candidates
        for seed in range(1, protocol.lr_sweep_seeds + 1)
    ]
    rows = parallel_runs(tasks, common, protocol.models_per_gpu, f"2x7 LR sweep {dataset}")
    candidates = []
    for lr in protocol.lr_candidates:
        values = np.asarray([row["test_accuracy"] for row in rows if row["lr"] == lr])
        candidates.append({"lr": lr, "test_accuracy": values.tolist(), "mean": float(values.mean()), "std": float(values.std()), "released_score_mean_minus_std": float(values.mean() - values.std())})
    chosen = sorted(candidates, key=lambda row: (-row["released_score_mean_minus_std"], row["lr"]))[0]
    payload = {
        "dataset": dataset,
        "dataset_sha256": sha256_file(dataset_pt),
        "protocol_sha256": hashlib.sha256(json.dumps(protocol.to_dict(), sort_keys=True).encode()).hexdigest(),
        "selection_uses_test_labels": True,
        "selection_protocol": "released WeightCLIP mean(test)-std(test); fidelity only",
        "candidates": candidates,
        "chosen_lr": chosen["lr"],
    }
    write_json_atomic(sweep_path, payload)
    print(f"[sweep:{dataset}] selected max_lr={chosen['lr']} -> {sweep_path}")
    return float(chosen["lr"])


def maybe_reuse_land_use_probe(
    config: dict[str, Any], dataset_pt: Path, output_root: Path, protocol: ZooProtocol
) -> None:
    """Reuse the completed official 2x7 sweep iff tensor payloads are identical."""

    target = output_root / "land_use" / "lr_sweep.json"
    if target.exists():
        return
    probe_root_raw = config.get("paths", {}).get("completed_land_use_probe")
    if not probe_root_raw:
        return
    probe_root = Path(probe_root_raw)
    summary = probe_root / "zoos/land-cover-class-0-10/resnet18slim_05x/tune_zoo_land-cover-class-0-10_resnet18slim_05x/lr_sweep/summary.json"
    probe_dataset = Path(config["paths"].get("completed_land_use_probe_dataset", ""))
    if not summary.exists() or not probe_dataset.exists():
        return
    left, right = load_dataset(dataset_pt), load_dataset(probe_dataset)
    identical = all(
        torch.equal(left[key].data, right[key].data) and torch.equal(left[key].targets, right[key].targets)
        for key in ("trainset", "valset", "testset")
    )
    if not identical:
        print("[sweep:land_use] completed probe NOT reused: materialized tensors differ")
        return
    source = json.loads(summary.read_text())
    payload = {
        "dataset": "land_use",
        "dataset_sha256": sha256_file(dataset_pt),
        "protocol_sha256": hashlib.sha256(json.dumps(protocol.to_dict(), sort_keys=True).encode()).hexdigest(),
        "selection_uses_test_labels": True,
        "selection_protocol": "released WeightCLIP mean(test)-std(test); fidelity only",
        "candidates": source["candidates"],
        "chosen_lr": float(source["chosen_lr"]),
        "reused_probe_summary": str(summary.resolve()),
        "probe_dataset_sha256": sha256_file(probe_dataset),
        "tensor_equality_verified": True,
    }
    write_json_atomic(target, payload)
    print(f"[sweep:land_use] reused completed official probe after exact tensor equality: {summary} -> {target}")


def _common(dataset: str, dataset_pt: Path, output_root: Path, protocol: ZooProtocol, verbose: bool) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_pt": str(dataset_pt),
        "dataset_sha256": sha256_file(dataset_pt),
        "output_root": str(output_root),
        "epochs": protocol.epochs,
        "scheduler_epochs": protocol.scheduler_epochs,
        "batch_size": protocol.batch_size,
        "momentum": protocol.momentum,
        "weight_decay": protocol.weight_decay,
        "width_mult": protocol.width_mult,
        "dropout": protocol.dropout,
        "init_type": protocol.init_type,
        "grad_clip": protocol.grad_clip,
        "verbose": verbose,
    }


def train_dataset(dataset: str, dataset_pt: Path, output_root: Path, protocol: ZooProtocol, chosen_lr: float, verbose: bool) -> None:
    common = _common(dataset, dataset_pt, output_root, protocol, verbose)
    tasks = []
    for seed in range(1, protocol.lineages_per_dataset + 1):
        complete = output_root / dataset / f"lineage-{seed:04d}" / "complete.json"
        if complete.exists():
            lineage_dir = complete.parent
            resolved = json.loads((lineage_dir / "resolved_config.json").read_text())
            expected = {**common, "seed": seed, "lr": chosen_lr, "save": True}
            for key in expected:
                if resolved.get(key) != expected.get(key):
                    raise ValueError(f"stale lineage cache {lineage_dir}: {key}={resolved.get(key)!r}, expected {expected.get(key)!r}")
            summary = json.loads(complete.read_text())
            summary_expected = {
                "dataset": dataset,
                "seed": seed,
                "lr": float(chosen_lr),
                "epoch_index_zero_based": protocol.epochs - 1,
                "epoch_one_based": protocol.epochs,
            }
            for key, value in summary_expected.items():
                if summary.get(key) != value:
                    raise ValueError(
                        f"stale lineage completion {complete}: {key}={summary.get(key)!r}, expected {value!r}"
                    )
            missing = [epoch for epoch in range(1, protocol.epochs + 1) if not (lineage_dir / "epochs" / f"epoch-{epoch:03d}.pt").is_file()]
            if missing or not (lineage_dir / "initialization.pt").is_file():
                raise ValueError(f"incomplete checkpoint inventory in {lineage_dir}; missing epochs={missing[:5]}")
            print(f"[train:{dataset}] cache hit seed={seed}: {complete}")
        else:
            tasks.append({"seed": seed, "lr": chosen_lr, "save": True})
    parallel_runs(tasks, common, protocol.models_per_gpu, f"final 50 lineages {dataset}")


def build_manifest(dataset_root: Path, output_root: Path, manifest_root: Path, protocol: ZooProtocol, *, audit_forward: bool, verbose: bool) -> dict[str, int]:
    config_hash = __import__("hashlib").sha256(json.dumps(protocol.to_dict(), sort_keys=True).encode()).hexdigest()
    lineages: list[LineageRecord] = []
    checkpoints: list[CheckpointRecord] = []
    for task in SOURCE_TASKS:
        dataset_pt = dataset_root / task.key / "dataset.pt"
        dataset_sha = sha256_file(dataset_pt)
        split_map = stable_lineage_split(task.key, range(1, 51), protocol.lineage_split, protocol.seed_namespace)
        dataset_payload = load_dataset(dataset_pt) if audit_forward else None
        classes = int(torch.unique(torch.cat([value.targets for value in dataset_payload.values()])).numel()) if dataset_payload else 0
        fixed_images = dataset_payload["trainset"].data[:8].cuda() if audit_forward else None
        for seed in range(1, 51):
            lineage_id = f"{task.key}:seed={seed}"
            lineage_dir = output_root / task.key / f"lineage-{seed:04d}"
            complete_path = lineage_dir / "complete.json"
            if not complete_path.exists():
                raise FileNotFoundError(f"incomplete lineage: {lineage_id}: {complete_path}")
            summary = json.loads(complete_path.read_text())
            resume = lineage_dir / "resume.pt"
            lineages.append(
                LineageRecord(
                    task.key,
                    lineage_id,
                    seed,
                    split_map[seed],
                    float(summary["lr"]),
                    dataset_sha,
                    config_hash,
                    "complete",
                    46,
                    str(resume.resolve()) if resume.exists() else None,
                    sha256_file(resume) if resume.exists() else None,
                )
            )
            metric_rows = {int(row["epoch_one_based"]): row for row in map(json.loads, (lineage_dir / "metrics.jsonl").read_text().splitlines())}
            archive_paths = [(0, -1, lineage_dir / "initialization.pt")]
            archive_paths.extend((epoch, epoch - 1, lineage_dir / "epochs" / f"epoch-{epoch:03d}.pt") for epoch in range(1, 46))
            for epoch_one, epoch_zero, path in archive_paths:
                if not path.exists():
                    raise FileNotFoundError(path)
                state = None
                if audit_forward:
                    state = torch.load(path, map_location="cuda", weights_only=True)
                    model = ResNet18Slim(3, classes, "relu", protocol.dropout, None, protocol.width_mult).cuda().eval()
                    model.load_state_dict(state, strict=True)
                    logits = model(fixed_images)
                    if not torch.isfinite(logits).all():
                        raise ValueError(f"nonfinite logits: {path}")
                metrics = metric_rows.get(epoch_one, {})
                checkpoints.append(
                    CheckpointRecord(
                        dataset=task.key,
                        lineage_id=lineage_id,
                        seed=seed,
                        split=split_map[seed],
                        epoch_one_based=epoch_one,
                        checkpoint_index_zero_based=epoch_zero,
                        is_primary=epoch_zero in protocol.primary_checkpoint_indices,
                        checkpoint_path=str(path.resolve()),
                        checkpoint_sha256=sha256_file(path),
                        bytes=path.stat().st_size,
                        train_loss=metrics.get("train_loss"),
                        train_accuracy=metrics.get("train_accuracy"),
                        validation_loss=metrics.get("validation_loss"),
                        validation_accuracy=metrics.get("validation_accuracy"),
                        test_loss=metrics.get("test_loss"),
                        test_accuracy=metrics.get("test_accuracy"),
                        chosen_lr=float(summary["lr"]),
                    )
                )
            if verbose:
                print(f"[manifest:{task.key}] audited lineage={seed}/50 split={split_map[seed]}")
    counts = validate_lineage_records(lineages, checkpoints, datasets=10, lineages_per_dataset=50)
    manifest_root.mkdir(parents=True, exist_ok=True)
    lineage_files = write_records_immutable(manifest_root / "lineage_manifest", lineages)
    checkpoint_files = write_records_immutable(manifest_root / "checkpoint_manifest", checkpoints)
    summary = {
        "format_version": 1,
        "counts": counts,
        "contract": frozen_contract(),
        "lineage_files": lineage_files,
        "checkpoint_files": checkpoint_files,
        "audit_forward_every_checkpoint": audit_forward,
    }
    write_json_immutable(manifest_root / "manifest.json", summary)
    print(f"[manifest] complete counts={counts} -> {manifest_root}")
    return counts


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("config must be a mapping")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("materialize", "materialize-ood", "sweep", "train", "manifest", "all"))
    parser.add_argument("--config", type=Path, default=WORKSPACE / "conf/weightclip_benchmark/zoo.yaml")
    parser.add_argument("--dataset", choices=[task.key for task in SOURCE_TASKS], default=None)
    parser.add_argument("--overwrite-datasets", action="store_true")
    parser.add_argument("--no-forward-audit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    protocol_values = dict(config.get("protocol", {}))
    for name in ("primary_checkpoint_indices", "lineage_split", "lr_candidates"):
        if name in protocol_values:
            protocol_values[name] = tuple(protocol_values[name])
    protocol = ZooProtocol(**protocol_values)
    protocol.validate()
    paths = config["paths"]
    archive = Path(paths["official_train_archive"])
    ood_archive = Path(paths["official_ood_archive"])
    dataset_root = Path(paths["dataset_pt_root"])
    ood_dataset_root = Path(paths["ood_dataset_pt_root"])
    cifar10_download_root = Path(paths["cifar10_download_root"])
    output_root = Path(paths["zoo_root"])
    manifest_root = Path(paths["manifest_root"])
    verbose = not args.quiet
    if not torch.cuda.is_available() and args.stage in ("sweep", "train", "all"):
        raise RuntimeError("CUDA is required for zoo training")
    print("[startup] WeightCLIP zoo reconstruction")
    print(f"[startup] resolved_config={json.dumps(redact_secrets(config), sort_keys=True)}")
    print(f"[startup] device={'cuda:0' if torch.cuda.is_available() else 'cpu'} dtype=float32 seed_namespace={protocol.seed_namespace}")
    print(f"[startup] cache_mode=resume output={output_root} manifest={manifest_root}")
    model = ResNet18Slim(3, 10, dropout=protocol.dropout, init_type=protocol.init_type, width_mult=protocol.width_mult)
    print(f"[startup] ResNet18Slim parameters={count_parameters(model)}")
    if args.stage in ("materialize", "all"):
        materialize_archive(archive, dataset_root, overwrite=args.overwrite_datasets, verbose=verbose)
    if args.stage == "materialize-ood":
        ood = config["ood_materialization"]
        materialize_ood_datasets(
            ood_archive,
            ood_dataset_root,
            cifar10_download_root,
            expected_sha256=str(ood["official_archive_sha256"]),
            validation_fraction=float(ood["cifar10_validation_fraction"]),
            split_seed=int(ood["cifar10_split_seed"]),
            download=bool(ood["cifar10_download"]),
            overwrite=args.overwrite_datasets,
            verbose=verbose,
        )
    tasks = [task for task in SOURCE_TASKS if args.dataset is None or task.key == args.dataset]
    if args.stage in ("sweep", "train", "all"):
        for index, task in enumerate(tasks, 1):
            dataset_pt = dataset_root / task.key / "dataset.pt"
            print(f"[dataset {index}/{len(tasks)}] {task.key} dataset_pt={dataset_pt} sha256={sha256_file(dataset_pt)}")
            if task.key == "land_use":
                maybe_reuse_land_use_probe(config, dataset_pt, output_root, protocol)
            chosen_lr = run_sweep(task.key, dataset_pt, output_root, protocol, verbose)
            if args.stage in ("train", "all"):
                train_dataset(task.key, dataset_pt, output_root, protocol, chosen_lr, verbose)
    if args.stage in ("manifest", "all"):
        build_manifest(dataset_root, output_root, manifest_root, protocol, audit_forward=not args.no_forward_audit, verbose=verbose)
    print(
        f"[done] artifacts: datasets={dataset_root} ood_datasets={ood_dataset_root} "
        f"zoo={output_root} manifests={manifest_root}"
    )


if __name__ == "__main__":
    main()
