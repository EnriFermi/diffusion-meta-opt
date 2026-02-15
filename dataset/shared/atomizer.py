from __future__ import annotations

import math
import random
import time
from typing import Iterator

from dataset.models.types import LayerIORecord
from dataset.shared.types import MixedImageMeta, SharedSample


def atomize(
    layer_record: LayerIORecord,
    atom_cfg: dict,
    image_meta_list: list[MixedImageMeta],
    model_run_id: int,
) -> Iterator[SharedSample]:
    """Convert one LayerIORecord into atomic SharedSample items."""

    slice_cfg = atom_cfg.get("xy_samples_random_slice")
    if slice_cfg is None:
        raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer")
    if isinstance(slice_cfg, str) and slice_cfg.strip().lower() in {"none", "null", ""}:
        raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer")
    try:
        slice_size = int(slice_cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer") from exc
    if slice_size <= 0:
        raise ValueError("collector.layer_output_splitting.xy_samples_random_slice must be a positive integer")

    inputs = layer_record.inputs
    outputs = layer_record.outputs

    if inputs.shape[0] != outputs.shape[0]:
        rows = min(inputs.shape[0], outputs.shape[0])
        inputs = inputs[:rows]
        outputs = outputs[:rows]

    num_rows = int(inputs.shape[0])
    row2img = _build_row2img(layer_record=layer_record, image_meta_list=image_meta_list, num_rows=num_rows)

    base_meta = {
        "model_run_id": int(model_run_id),
        "timestamp": time.time(),
        "image_meta": [
            {"dataset_name": item.dataset_name, "source_id": item.source_id}
            for item in image_meta_list
        ],
        "layer_meta": dict(layer_record.meta),
    }

    if num_rows <= 0:
        local_meta = dict(base_meta)
        local_meta["xy_sampling_mode"] = "full"
        local_meta["selected_row_count"] = 0
        local_meta["selected_row_indices_preview"] = []
        local_meta["row_start"] = 0
        local_meta["row_end"] = 0
        if row2img is not None:
            local_meta["row2img"] = []
        yield SharedSample(
            model_name=layer_record.model_name,
            layer_name=layer_record.layer_name,
            weight=layer_record.weight,
            x=inputs,
            y=outputs,
            meta=local_meta,
        )
        return

    if num_rows <= slice_size:
        selected_indices = list(range(num_rows))
        sampling_mode = "full"
    else:
        selected_indices = sorted(random.sample(range(num_rows), k=slice_size))
        sampling_mode = "random_slice"

    x_selected = inputs[selected_indices]
    y_selected = outputs[selected_indices]

    local_meta = dict(base_meta)
    local_meta["xy_sampling_mode"] = sampling_mode
    local_meta["selected_row_count"] = len(selected_indices)
    local_meta["selected_row_indices_preview"] = selected_indices[:32]
    local_meta["row_start"] = 0
    local_meta["row_end"] = len(selected_indices)
    if row2img is not None:
        local_meta["row2img"] = [row2img[idx] for idx in selected_indices]

    yield SharedSample(
        model_name=layer_record.model_name,
        layer_name=layer_record.layer_name,
        weight=layer_record.weight,
        x=x_selected,
        y=y_selected,
        meta=local_meta,
    )


def _build_row2img(
    layer_record: LayerIORecord,
    image_meta_list: list[MixedImageMeta],
    num_rows: int,
) -> list[int] | None:
    if not image_meta_list:
        return None

    if num_rows <= 0:
        return []

    input_shapes = layer_record.meta.get("input_shape_list", []) if isinstance(layer_record.meta, dict) else []

    if not input_shapes:
        if num_rows == len(image_meta_list):
            return list(range(num_rows))
        return None

    row2img: list[int] = []
    image_cursor = 0

    for raw_shape in input_shapes:
        if not isinstance(raw_shape, (tuple, list)):
            continue
        shape = tuple(int(item) for item in raw_shape)
        if len(shape) < 1:
            continue

        batch = shape[0]
        if batch <= 0:
            continue

        if len(shape) <= 2:
            rows_per_image = 1
        else:
            rows_per_image = int(math.prod(shape[1:-1]))
            rows_per_image = max(1, rows_per_image)

        for offset in range(batch):
            image_idx = (image_cursor + offset) % len(image_meta_list)
            row2img.extend([image_idx] * rows_per_image)

        image_cursor += batch

    if not row2img:
        return None

    if len(row2img) < num_rows:
        pad_value = row2img[-1]
        row2img.extend([pad_value] * (num_rows - len(row2img)))

    return row2img[:num_rows]
