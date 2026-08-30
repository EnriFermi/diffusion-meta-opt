from __future__ import annotations

import math
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Any

import numpy as np
import torch
from PIL import Image

from dataset.models.types import LayerIORecord
from dataset.shared.types import SharedSample


@dataclass(slots=True)
class SharedBufferRef:
    name: str
    size_bytes: int


@dataclass(slots=True)
class SharedImageRef:
    buffer: SharedBufferRef
    mode: str
    size: tuple[int, int]


@dataclass(slots=True)
class SharedTensorRef:
    buffer: SharedBufferRef
    shape: tuple[int, ...]
    dtype: str


@dataclass(slots=True)
class SharedLayerRecordRef:
    model_name: str
    layer_name: str
    weight: SharedTensorRef
    inputs: SharedTensorRef
    outputs: SharedTensorRef
    meta: dict[str, Any]


@dataclass(slots=True)
class SharedSampleRef:
    model_name: str
    layer_name: str
    weight: SharedTensorRef
    x: SharedTensorRef
    y: SharedTensorRef
    meta: dict[str, Any]


def _resource_tracker_name(shm: shared_memory.SharedMemory | str) -> str:
    if isinstance(shm, str):
        return shm
    return str(getattr(shm, "_name", shm.name))


def _best_effort_unregister_shared_memory(name: str) -> None:
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass


def _create_shared_buffer(raw_bytes: bytes) -> SharedBufferRef:
    actual_size = int(len(raw_bytes))
    shm = shared_memory.SharedMemory(create=True, size=max(1, actual_size))
    try:
        if actual_size > 0:
            shm.buf[:actual_size] = raw_bytes
        return SharedBufferRef(name=str(shm.name), size_bytes=actual_size)
    finally:
        _best_effort_unregister_shared_memory(_resource_tracker_name(shm))
        shm.close()


def _create_shared_ndarray_copy(array: np.ndarray) -> SharedBufferRef:
    contiguous = np.ascontiguousarray(array)
    actual_size = int(contiguous.nbytes)
    shm = shared_memory.SharedMemory(create=True, size=max(1, actual_size))
    try:
        if actual_size > 0:
            shm_array = np.ndarray(shape=contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
            np.copyto(shm_array, contiguous, casting="no")
        return SharedBufferRef(name=str(shm.name), size_bytes=actual_size)
    finally:
        _best_effort_unregister_shared_memory(_resource_tracker_name(shm))
        shm.close()


def release_shared_buffer(ref: SharedBufferRef) -> None:
    shm = shared_memory.SharedMemory(name=str(ref.name))
    try:
        shm.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def share_image(image: Image.Image) -> SharedImageRef:
    materialized = image.copy()
    materialized.load()
    image_array = np.asarray(materialized)
    return SharedImageRef(
        buffer=_create_shared_ndarray_copy(image_array),
        mode=str(materialized.mode),
        size=(int(materialized.size[0]), int(materialized.size[1])),
    )


def restore_image(ref: SharedImageRef, *, release: bool = True) -> Image.Image:
    shm = shared_memory.SharedMemory(name=str(ref.buffer.name))
    try:
        raw_bytes = bytes(shm.buf[: int(ref.buffer.size_bytes)])
    finally:
        shm.close()
        if release:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
    image = Image.frombytes(str(ref.mode), tuple(ref.size), raw_bytes)
    image.load()
    return image


def share_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shared_rows: list[dict[str, Any]] = []
    try:
        for row in rows:
            image = row.get("image")
            if not isinstance(image, Image.Image):
                continue
            shared_rows.append(
                {
                    "sample_id": row.get("sample_id", "unknown"),
                    "meta": row.get("meta", {}) if isinstance(row.get("meta"), dict) else {},
                    "image_ref": share_image(image),
                }
            )
        return shared_rows
    except Exception:
        cleanup_shared_image_rows(shared_rows)
        raise


def restore_image_rows(rows: list[dict[str, Any]], *, release: bool = True) -> list[dict[str, Any]]:
    restored_rows: list[dict[str, Any]] = []
    try:
        for row in rows:
            image_ref = row.get("image_ref")
            if not isinstance(image_ref, SharedImageRef):
                continue
            restored_rows.append(
                {
                    "sample_id": row.get("sample_id", "unknown"),
                    "meta": row.get("meta", {}) if isinstance(row.get("meta"), dict) else {},
                    "image": restore_image(image_ref, release=release),
                }
            )
        return restored_rows
    except Exception:
        if release:
            cleanup_shared_image_rows(rows)
        raise


def cleanup_shared_image_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        image_ref = row.get("image_ref")
        if not isinstance(image_ref, SharedImageRef):
            continue
        try:
            release_shared_buffer(image_ref.buffer)
        except Exception:
            pass


def share_tensor(tensor: torch.Tensor) -> SharedTensorRef:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    array = cpu_tensor.numpy()
    return SharedTensorRef(
        buffer=_create_shared_ndarray_copy(array),
        shape=tuple(int(dim) for dim in array.shape),
        dtype=str(array.dtype.str),
    )


def restore_tensor(ref: SharedTensorRef, *, release: bool = True) -> torch.Tensor:
    dtype = np.dtype(str(ref.dtype))
    numel = int(math.prod(ref.shape)) if ref.shape else 1
    if int(ref.buffer.size_bytes) <= 0 or numel <= 0:
        empty_array = np.empty(shape=tuple(ref.shape), dtype=dtype)
        return torch.from_numpy(empty_array).clone()

    shm = shared_memory.SharedMemory(name=str(ref.buffer.name))
    try:
        array = np.ndarray(shape=tuple(ref.shape), dtype=dtype, buffer=shm.buf)
        tensor = torch.from_numpy(np.array(array, copy=True))
    finally:
        shm.close()
        if release:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
    return tensor


def share_layer_record_refs(layer_records: list[LayerIORecord]) -> list[SharedLayerRecordRef]:
    shared_records: list[SharedLayerRecordRef] = []
    try:
        for record in layer_records:
            shared_records.append(
                SharedLayerRecordRef(
                    model_name=str(record.model_name),
                    layer_name=str(record.layer_name),
                    weight=share_tensor(record.weight),
                    inputs=share_tensor(record.inputs),
                    outputs=share_tensor(record.outputs),
                    meta=dict(record.meta),
                )
            )
        return shared_records
    except Exception:
        cleanup_shared_layer_record_refs(shared_records)
        raise


def restore_layer_record_refs(
    record_refs: list[SharedLayerRecordRef],
    *,
    release: bool = True,
) -> list[LayerIORecord]:
    restored_records: list[LayerIORecord] = []
    try:
        for record_ref in record_refs:
            restored_records.append(
                LayerIORecord(
                    model_name=str(record_ref.model_name),
                    layer_name=str(record_ref.layer_name),
                    weight=restore_tensor(record_ref.weight, release=release),
                    inputs=restore_tensor(record_ref.inputs, release=release),
                    outputs=restore_tensor(record_ref.outputs, release=release),
                    meta=dict(record_ref.meta),
                )
            )
        return restored_records
    except Exception:
        if release:
            cleanup_shared_layer_record_refs(record_refs)
        raise


def cleanup_shared_layer_record_refs(record_refs: list[SharedLayerRecordRef]) -> None:
    for record_ref in record_refs:
        for tensor_ref in (record_ref.weight, record_ref.inputs, record_ref.outputs):
            try:
                release_shared_buffer(tensor_ref.buffer)
            except Exception:
                pass


def share_shared_sample(sample: SharedSample) -> SharedSampleRef:
    shared_ref: SharedSampleRef | None = None
    try:
        shared_ref = SharedSampleRef(
            model_name=str(sample.model_name),
            layer_name=str(sample.layer_name),
            weight=share_tensor(sample.weight),
            x=share_tensor(sample.x),
            y=share_tensor(sample.y),
            meta=dict(sample.meta),
        )
        return shared_ref
    except Exception:
        if shared_ref is not None:
            cleanup_shared_sample_ref(shared_ref)
        raise


def restore_shared_sample(ref: SharedSampleRef, *, release: bool = True) -> SharedSample:
    try:
        return SharedSample(
            model_name=str(ref.model_name),
            layer_name=str(ref.layer_name),
            weight=restore_tensor(ref.weight, release=release),
            x=restore_tensor(ref.x, release=release),
            y=restore_tensor(ref.y, release=release),
            meta=dict(ref.meta),
        )
    except Exception:
        if release:
            cleanup_shared_sample_ref(ref)
        raise


def cleanup_shared_sample_ref(ref: SharedSampleRef) -> None:
    for tensor_ref in (ref.weight, ref.x, ref.y):
        try:
            release_shared_buffer(tensor_ref.buffer)
        except Exception:
            pass
