from __future__ import annotations

import math
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

    atom_mode = str(atom_cfg.get("sample_granularity", "chunk")).lower()
    chunk_rows = max(1, int(atom_cfg.get("rows_per_chunk_sample", 256)))

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

    if atom_mode == "row":
        for row_idx in range(num_rows):
            local_meta = dict(base_meta)
            local_meta["row_start"] = row_idx
            local_meta["row_end"] = row_idx + 1
            if row2img is not None and row_idx < len(row2img):
                local_meta["row2img"] = [row2img[row_idx]]

            yield SharedSample(
                model_name=layer_record.model_name,
                layer_name=layer_record.layer_name,
                weight=layer_record.weight,
                x=inputs[row_idx : row_idx + 1],
                y=outputs[row_idx : row_idx + 1],
                meta=local_meta,
            )
        return

    for start in range(0, num_rows, chunk_rows):
        end = min(num_rows, start + chunk_rows)
        local_meta = dict(base_meta)
        local_meta["row_start"] = start
        local_meta["row_end"] = end
        if row2img is not None:
            local_meta["row2img"] = row2img[start:end]

        yield SharedSample(
            model_name=layer_record.model_name,
            layer_name=layer_record.layer_name,
            weight=layer_record.weight,
            x=inputs[start:end],
            y=outputs[start:end],
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
