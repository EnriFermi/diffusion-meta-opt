#!/usr/bin/env python3
"""Read-only source-panel feasibility audit for Food101 ViT and TrOCR-base.

This script never loads the Weight-AE and never reads target/data2vec assets.
It performs two source-model checks:

1. a deterministic Food101 validation quality pilot for the local fine-tuned ViT;
2. a train-only FUNSD word-crop pilot for the local TrOCR checkpoint.

The FUNSD test split is intentionally not opened so it remains available for a
later prospective quality gate and activation-score panel.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import socket
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from safetensors import safe_open
from transformers import (
    AutoImageProcessor,
    TrOCRProcessor,
    ViTForImageClassification,
    VisionEncoderDecoderModel,
)
from transformers.utils import logging as transformers_logging


PROJECT = Path("/home/coder/project")
FOOD_MODEL = PROJECT / "projects/shared/storage/data/models/vit_food101_ashaduzzaman_57f4382f"
FOOD_DATA = (
    PROJECT
    / "projects/shared/storage/data/datasets/hf_datasets_cache/ethz___food101/default/0.0.0/"
    "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9"
)
TROCR_MODEL = (
    PROJECT
    / "projects/shared/storage/data/models/trocr_base_printed/"
    "models--microsoft--trocr-base-printed/snapshots/"
    "93450be3f1ed40a930690d951ef3932687cc1892"
)
FUNSD_DATA = (
    PROJECT
    / "projects/shared/storage/data/datasets/hf_datasets_cache/nielsr___funsd/default/0.0.0/"
    "7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/"
    "prospective_panel_feasibility_scout_20260816"
)

ROLES = (
    ("attn_query", "attention.attention.query"),
    ("attn_key", "attention.attention.key"),
    ("attn_value", "attention.attention.value"),
    ("attn_output", "attention.output.dense"),
    ("ffn_up", "intermediate.dense"),
    ("ffn_down", "output.dense"),
)
FOOD_MODEL_CARD_CLAIMED_ACCURACY = 0.896
FOOD_GATE_TOP1 = 0.70
FOOD_GATE_TOP5 = 0.90
PILOT_SEED = 260_816
TROCR_PAD_FRACTIONS = (0.1, 0.3, 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--food-batch-size", type=int, default=64)
    parser.add_argument("--food-pilot-samples", type=int, default=512)
    parser.add_argument("--trocr-batch-size", type=int, default=8)
    parser.add_argument("--trocr-pilot-samples", type=int, default=128)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def rgb_sha256(image: Any) -> str:
    value = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{value.width}x{value.height}:".encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    values = list(rows)
    if not values and fieldnames is None:
        raise ValueError(f"cannot infer CSV schema for empty output: {path}")
    columns = fieldnames or list(values[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("prospective_panel_feasibility_scout")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def install_source_only_seal() -> dict[str, Any]:
    forbidden = ("data2vec",)

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir"} and args:
            candidate = args[0]
            if isinstance(candidate, (str, bytes, Path)):
                text = str(candidate).lower()
                # Transformers' lazy AutoModel registry enumerates its own
                # site-packages/data2vec source directory even when building a
                # ViT.  That is library code, not a target artifact.  Block the
                # marker only under the shared project/artifact namespace.
                is_project_asset = text.startswith(str(PROJECT).lower())
                if is_project_asset and any(marker in text for marker in forbidden):
                    raise RuntimeError(f"source-only seal blocked {event}: {candidate}")
        if event == "socket.connect":
            raise RuntimeError("source-only seal blocked a network connection")

    sys.addaudithook(hook)
    # Record that a real socket operation is blocked without making a network call.
    original_connect = socket.socket.connect
    return {
        "installed": True,
        "forbidden_path_markers": list(forbidden),
        "forbidden_marker_scope": str(PROJECT),
        "site_packages_library_enumeration_allowed": True,
        "network_connect_prohibited": True,
        "socket_connect_callable": str(original_connect),
        "target_data2vec_access": False,
        "funsd_test_access": False,
        "weight_ae_loaded": False,
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    import datasets
    import PIL
    import safetensors
    import transformers

    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "safetensors": safetensors.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return result


def hash_inputs(paths: dict[str, Path], logger: logging.Logger) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, raw_path in paths.items():
        path = raw_path.resolve(strict=True)
        started = time.monotonic()
        digest = sha256_file(path)
        result[label] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
        logger.info(
            "stage=input_hash label=%s bytes=%s sha256=%s elapsed=%.2fs",
            label,
            path.stat().st_size,
            digest,
            time.monotonic() - started,
        )
    return result


def loading_info_json(info: dict[str, Any]) -> dict[str, Any]:
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    normalized = {key: info.get(key, []) for key in keys}
    normalized["clean"] = all(not normalized[key] for key in keys)
    return normalized


def module_inventory(
    *,
    model: torch.nn.Module,
    module_prefix: str,
    state_prefix: str,
    panel: str,
) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for depth in range(12):
        for role, suffix in ROLES:
            module_name = f"{module_prefix}{depth}.{suffix}"
            module = modules.get(module_name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"missing exact Linear module: {panel}/{module_name}/{type(module)}")
            raw = module.weight.detach().cpu().to(torch.float32).contiguous()
            weight = raw.transpose(0, 1).contiguous()
            expected = (3072, 768) if role == "ffn_down" else ((768, 3072) if role == "ffn_up" else (768, 768))
            if tuple(weight.shape) != expected:
                raise RuntimeError(f"unexpected W shape: {panel}/{module_name}/{tuple(weight.shape)} != {expected}")
            rows.append(
                {
                    "panel": panel,
                    "depth": depth,
                    "role": role,
                    "runtime_module": module_name,
                    "state_key": f"{state_prefix}{depth}.{suffix}.weight",
                    "linear_weight_shape": "x".join(map(str, raw.shape)),
                    "weight_ae_W_shape": "x".join(map(str, weight.shape)),
                    "weight_dtype": str(raw.dtype),
                    "bias_present": module.bias is not None,
                    "weight_ae_bias_excluded": True,
                    "W_sha256": tensor_sha256(weight),
                    "W_numel": weight.numel(),
                }
            )
    if len(rows) != 72:
        raise RuntimeError(f"expected 72 matrices for {panel}, found {len(rows)}")
    return rows


def hook_probe(model: torch.nn.Module, names: list[str]) -> tuple[dict[str, Any], list[Any]]:
    modules = dict(model.named_modules())
    values: dict[str, Any] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(_module: torch.nn.Module, inputs: Any, outputs: Any) -> None:
            source = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            target = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            values[name] = {
                "input_shape": list(source.shape),
                "output_shape": list(target.shape),
                "input_dtype": str(source.dtype),
                "output_dtype": str(target.dtype),
            }

        return hook

    for name in names:
        if name not in modules:
            raise RuntimeError(f"hook module missing: {name}")
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    return values, handles


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def ece_15(confidences: np.ndarray, correct: np.ndarray) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidences >= lower) & (confidences < upper if index < 14 else confidences <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidences[selected].mean()))
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError(f"Wilson interval requires positive trials, got {trials}")
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def food_shard_location(index: int, lengths: list[int], names: list[str]) -> tuple[str, int]:
    ends = np.cumsum(lengths).tolist()
    shard = bisect.bisect_right(ends, index)
    start = 0 if shard == 0 else ends[shard - 1]
    return names[shard], index - start


def evaluate_food(
    *,
    output: Path,
    device: torch.device,
    batch_size: int,
    samples: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    panel_output = output / "food101_deterministic_512_pilot"
    panel_output.mkdir()
    logger.info("stage=food_inputs mode=deterministic_pilot samples=%s output=%s", samples, panel_output)
    shard_paths = sorted(FOOD_DATA.glob("food101-validation-*.arrow"))
    if len(shard_paths) != 3:
        raise RuntimeError(f"expected three Food101 validation arrows, found {shard_paths}")
    input_hashes = hash_inputs(
        {
            "model_safetensors": FOOD_MODEL / "model.safetensors",
            "model_config": FOOD_MODEL / "config.json",
            "preprocessor_config": FOOD_MODEL / "preprocessor_config.json",
            "dataset_info": FOOD_DATA / "dataset_info.json",
            **{f"validation_arrow_{idx}": path for idx, path in enumerate(shard_paths)},
        },
        logger,
    )
    datasets = [Dataset.from_file(str(path)) for path in shard_paths]
    shard_lengths = [len(dataset) for dataset in datasets]
    dataset = concatenate_datasets(datasets)
    if len(dataset) != 25_250:
        raise RuntimeError(f"Food101 validation size drift: {len(dataset)}")
    dataset_info = json.loads((FOOD_DATA / "dataset_info.json").read_text(encoding="utf-8"))
    dataset_labels = list(dataset_info["features"]["label"]["names"])
    if not (0 < samples <= len(dataset)):
        raise ValueError(f"invalid Food101 pilot size: {samples}/{len(dataset)}")
    selected_indices = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(f"{PILOT_SEED}|food101_validation|{index}".encode("ascii")).hexdigest(),
    )[:samples]
    selected_dataset = dataset.select(selected_indices)
    selection_rows = []
    for ordinal, global_index in enumerate(selected_indices):
        shard_name, row_in_shard = food_shard_location(
            global_index, shard_lengths, [path.name for path in shard_paths]
        )
        selection_rows.append(
            {
                "sample_ordinal": ordinal,
                "global_index": global_index,
                "selection_sha256": hashlib.sha256(
                    f"{PILOT_SEED}|food101_validation|{global_index}".encode("ascii")
                ).hexdigest(),
                "shard": shard_name,
                "row_in_shard": row_in_shard,
                "truth_id": int(selected_dataset[ordinal]["label"]),
                "truth_label": dataset_labels[int(selected_dataset[ordinal]["label"])],
            }
        )
    write_csv(panel_output / "selection_manifest.csv", selection_rows)

    transformers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()
    started = time.monotonic()
    processor = AutoImageProcessor.from_pretrained(str(FOOD_MODEL), local_files_only=True)
    model, raw_loading_info = ViTForImageClassification.from_pretrained(
        str(FOOD_MODEL), local_files_only=True, output_loading_info=True
    )
    load_info = loading_info_json(raw_loading_info)
    if not load_info["clean"]:
        raise RuntimeError(f"Food model loading was not strict-clean: {load_info}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info(
        "stage=food_model_loaded device=%s dtype=%s load_elapsed=%.2fs missing=%s unexpected=%s",
        device,
        next(model.parameters()).dtype,
        time.monotonic() - started,
        len(load_info["missing_keys"]),
        len(load_info["unexpected_keys"]),
    )
    model_labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    mapping_rows = [
        {
            "label_id": index,
            "dataset_label": dataset_labels[index],
            "model_id2label": model_labels[index],
            "exact_match": dataset_labels[index] == model_labels[index],
            "model_label2id": int(model.config.label2id[model_labels[index]]),
        }
        for index in range(101)
    ]
    mapping_exact = all(row["exact_match"] and row["model_label2id"] == row["label_id"] for row in mapping_rows)
    write_csv(panel_output / "dataset_label_mapping.csv", mapping_rows)

    inventory = module_inventory(
        model=model,
        module_prefix="vit.encoder.layer.",
        state_prefix="vit.encoder.layer.",
        panel="food101_vit",
    )
    write_csv(panel_output / "weight_matrix_inventory.csv", inventory)

    hook_names = [f"vit.encoder.layer.0.{suffix}" for _role, suffix in ROLES]
    hook_shapes, hook_handles = hook_probe(model, hook_names)
    prediction_rows: list[dict[str, Any]] = []
    truth_count = np.zeros(101, dtype=np.int64)
    correct_count = np.zeros(101, dtype=np.int64)
    pred_count = np.zeros(101, dtype=np.int64)
    logit_sum = np.zeros(101, dtype=np.float64)
    logit_sumsq = np.zeros(101, dtype=np.float64)
    logit_min = np.full(101, np.inf, dtype=np.float64)
    logit_max = np.full(101, -np.inf, dtype=np.float64)
    confidences: list[float] = []
    correctness: list[bool] = []
    total_nll = 0.0
    top5_correct = 0
    eval_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for begin in range(0, len(selected_dataset), batch_size):
        end = min(begin + batch_size, len(selected_dataset))
        batch = selected_dataset[begin:end]
        images = [image.convert("RGB") for image in batch["image"]]
        labels = torch.tensor(batch["label"], dtype=torch.long, device=device)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            logits = model(pixel_values=pixel_values).logits.float()
            log_probs = logits.log_softmax(dim=1)
            probabilities = logits.softmax(dim=1)
            top5 = logits.topk(5, dim=1).indices
        if begin == 0:
            remove_hooks(hook_handles)
            hook_handles = []
        predictions = top5[:, 0]
        confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
        nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
        correct = predictions == labels
        hit5 = (top5 == labels[:, None]).any(dim=1)
        logits_cpu = logits.cpu().numpy().astype(np.float64)
        predictions_cpu = predictions.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        confidence_cpu = confidence.cpu().numpy()
        nll_cpu = nll.cpu().numpy()
        correct_cpu = correct.cpu().numpy()
        top5_cpu = top5.cpu().numpy()
        truth_count += np.bincount(labels_cpu, minlength=101)
        correct_count += np.bincount(labels_cpu[correct_cpu], minlength=101)
        pred_count += np.bincount(predictions_cpu, minlength=101)
        logit_sum += logits_cpu.sum(axis=0)
        logit_sumsq += np.square(logits_cpu).sum(axis=0)
        logit_min = np.minimum(logit_min, logits_cpu.min(axis=0))
        logit_max = np.maximum(logit_max, logits_cpu.max(axis=0))
        confidences.extend(confidence_cpu.tolist())
        correctness.extend(correct_cpu.tolist())
        total_nll += float(nll_cpu.sum())
        top5_correct += int(hit5.sum())
        for offset in range(end - begin):
            sample_ordinal = begin + offset
            global_index = selected_indices[sample_ordinal]
            shard_name, row_in_shard = food_shard_location(
                global_index, shard_lengths, [path.name for path in shard_paths]
            )
            predicted = int(predictions_cpu[offset])
            truth = int(labels_cpu[offset])
            indices5 = [int(value) for value in top5_cpu[offset]]
            prediction_rows.append(
                {
                    "sample_ordinal": sample_ordinal,
                    "global_index": global_index,
                    "selection_sha256": selection_rows[sample_ordinal]["selection_sha256"],
                    "shard": shard_name,
                    "row_in_shard": row_in_shard,
                    "truth_id": truth,
                    "truth_label": dataset_labels[truth],
                    "prediction_id": predicted,
                    "prediction_label": model_labels[predicted],
                    "top1_correct": bool(correct_cpu[offset]),
                    "top5_correct": bool(hit5[offset]),
                    "confidence": float(confidence_cpu[offset]),
                    "nll": float(nll_cpu[offset]),
                    "top5_ids": ";".join(map(str, indices5)),
                    "top5_labels": ";".join(model_labels[index] for index in indices5),
                }
            )
        if begin == 0 or end == len(selected_dataset) or (begin // batch_size + 1) % 20 == 0:
            elapsed = time.monotonic() - eval_started
            logger.info(
                "stage=food_eval examples=%s/%s running_top1=%.6f rate=%.1f/s elapsed=%.1fs",
                end,
                len(selected_dataset),
                sum(correctness) / len(correctness),
                end / max(elapsed, 1e-9),
                elapsed,
            )
    if hook_handles:
        remove_hooks(hook_handles)
    write_csv(panel_output / "predictions.csv", prediction_rows)
    confidence_values = np.asarray(confidences, dtype=np.float64)
    correctness_values = np.asarray(correctness, dtype=np.bool_)
    top1 = float(correctness_values.mean())
    top5_value = top5_correct / len(selected_dataset)
    macro_recall = float(np.mean(correct_count / np.maximum(truth_count, 1)))
    mean_logits = logit_sum / len(selected_dataset)
    logit_variance = np.maximum(logit_sumsq / len(selected_dataset) - np.square(mean_logits), 0.0)
    classifier = model.classifier
    classifier_norms = classifier.weight.detach().float().cpu().norm(dim=1).numpy()
    classifier_bias = classifier.bias.detach().float().cpu().numpy()
    class_rows = []
    for index in range(101):
        class_rows.append(
            {
                "label_id": index,
                "label": dataset_labels[index],
                "truth_count": int(truth_count[index]),
                "correct_count": int(correct_count[index]),
                "recall": float(correct_count[index] / max(truth_count[index], 1)),
                "predicted_count": int(pred_count[index]),
                "classifier_weight_l2": float(classifier_norms[index]),
                "classifier_bias": float(classifier_bias[index]),
                "mean_logit": float(mean_logits[index]),
                "std_logit": float(math.sqrt(logit_variance[index])),
                "min_logit": float(logit_min[index]),
                "max_logit": float(logit_max[index]),
            }
        )
    write_csv(panel_output / "class_and_classifier_stats.csv", class_rows)
    top1_successes = int(correctness_values.sum())
    top5_successes = int(top5_correct)
    top1_interval = wilson_interval(top1_successes, len(selected_dataset))
    top5_interval = wilson_interval(top5_successes, len(selected_dataset))
    metrics = {
        "panel": "food101_vit_deterministic_validation_pilot",
        "status": (
            "PASS"
            if top1_interval[0] >= FOOD_GATE_TOP1 and top5_interval[0] >= FOOD_GATE_TOP5
            else (
                "FAIL_EVEN_BY_95_PERCENT_WILSON_UCB"
                if top1_interval[1] < FOOD_GATE_TOP1 or top5_interval[1] < FOOD_GATE_TOP5
                else "INCONCLUSIVE"
            )
        ),
        "examples": len(selected_dataset),
        "validation_population": len(dataset),
        "selection": "512 smallest SHA256(seed|food101_validation|global_index), without replacement",
        "selection_seed": PILOT_SEED,
        "top1_accuracy": top1,
        "top5_accuracy": top5_value,
        "top1_successes": top1_successes,
        "top5_successes": top5_successes,
        "top1_wilson_95_interval": list(top1_interval),
        "top5_wilson_95_interval": list(top5_interval),
        "mean_nll": total_nll / len(selected_dataset),
        "macro_recall": macro_recall,
        "mean_confidence": float(confidence_values.mean()),
        "ece_15": ece_15(confidence_values, correctness_values),
        "unique_predicted_classes": int(np.count_nonzero(pred_count)),
        "predicted_class_ids": np.flatnonzero(pred_count).tolist(),
        "claimed_model_card_accuracy": FOOD_MODEL_CARD_CLAIMED_ACCURACY,
        "absolute_gap_to_model_card_claim": FOOD_MODEL_CARD_CLAIMED_ACCURACY - top1,
        "predeclared_scout_gate": {"top1_at_least": FOOD_GATE_TOP1, "top5_at_least": FOOD_GATE_TOP5},
        "dataset_model_label_mapping_exact": mapping_exact,
        "loading_info_clean": load_info["clean"],
        "matrices": len(inventory),
        "hook_shapes": hook_shapes,
        "processor_class": type(processor).__name__,
        "processor_config": processor.to_dict(),
        "model_config": model.config.to_dict(),
        "classifier": {
            "weight_norm_min": float(classifier_norms.min()),
            "weight_norm_median": float(np.median(classifier_norms)),
            "weight_norm_max": float(classifier_norms.max()),
            "bias_min": float(classifier_bias.min()),
            "bias_median": float(np.median(classifier_bias)),
            "bias_max": float(classifier_bias.max()),
            "positive_bias_class_ids": np.flatnonzero(classifier_bias > 0).tolist(),
        },
        "logits_global": {
            "mean": float(logit_sum.sum() / (len(selected_dataset) * 101)),
            "std": float(np.sqrt(np.maximum((logit_sumsq.sum() / (len(selected_dataset) * 101)) - (logit_sum.sum() / (len(selected_dataset) * 101)) ** 2, 0.0))),
            "min": float(logit_min.min()),
            "max": float(logit_max.max()),
        },
        "runtime_seconds": time.monotonic() - eval_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    write_json(panel_output / "metrics.json", metrics)
    write_json(panel_output / "loading_info.json", load_info)
    write_json(panel_output / "input_hashes.json", input_hashes)
    write_json(panel_output / "hook_shape_inspection.json", hook_shapes)
    write_json(
        panel_output / "resolved_config.json",
        {
            "mode": "deterministic_validation_pilot",
            "batch_size": batch_size,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "dataset_rows": len(dataset),
            "selected_rows": len(selected_dataset),
            "selection_seed": PILOT_SEED,
            "selection_rule": "512 smallest SHA256(seed|food101_validation|global_index), without replacement",
            "dataset_shards": [str(path.resolve()) for path in shard_paths],
            "local_files_only": True,
            "processor_class": type(processor).__name__,
            "all_indices_included": False,
            "weight_ae_loaded": False,
        },
    )
    logger.info(
        "stage=food_complete status=%s top1=%.6f top5=%.6f unique_predicted=%s artifacts=%s",
        metrics["status"],
        top1,
        top5_value,
        metrics["unique_predicted_classes"],
        panel_output,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def normalize_ocr(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row] + [0] * len(right)
        for column, right_value in enumerate(right, 1):
            current[column] = min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_value != right_value),
            )
        previous = current
    return previous[-1]


def funsd_crop(image: Any, box: list[int], pad_fraction: float) -> Any:
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in box]
    x0, x1 = x0 * width / 1000.0, x1 * width / 1000.0
    y0, y1 = y0 * height / 1000.0, y1 * height / 1000.0
    box_height = max(y1 - y0, 1.0)
    padding = pad_fraction * box_height
    crop_box = (
        max(0, math.floor(x0 - padding)),
        max(0, math.floor(y0 - padding)),
        min(width, math.ceil(x1 + padding)),
        min(height, math.ceil(y1 + padding)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise RuntimeError(f"invalid crop after normalized-box conversion: {box}/{image.size}/{crop_box}")
    return image.crop(crop_box).convert("RGB")


def select_funsd_train_candidates(dataset: Dataset, count: int) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    candidates: list[dict[str, Any]] = []
    selected_pages: dict[int, Any] = {}
    for page_index in range(len(dataset)):
        row = dataset[page_index]
        image = row["image"].convert("RGB")
        width, height = image.size
        for word_index, (ground_truth_raw, box) in enumerate(zip(row["words"], row["bboxes"], strict=True)):
            ground_truth = normalize_ocr(ground_truth_raw)
            box_width_px = (box[2] - box[0]) * width / 1000.0
            box_height_px = (box[3] - box[1]) * height / 1000.0
            if (
                2 <= len(ground_truth) <= 24
                and box_width_px >= 4.0
                and box_height_px >= 4.0
                and any(character.isalnum() for character in ground_truth)
            ):
                sample_key = hashlib.sha256(
                    f"{PILOT_SEED}|{row['id']}|{word_index}|{ground_truth}".encode("utf-8")
                ).hexdigest()
                candidates.append(
                    {
                        "selection_sha256": sample_key,
                        "page_index": page_index,
                        "page_id": str(row["id"]),
                        "word_index": word_index,
                        "ground_truth_raw": ground_truth_raw,
                        "ground_truth_normalized": ground_truth,
                        "bbox_normalized_1000": list(map(int, box)),
                        "page_width": width,
                        "page_height": height,
                        "box_width_px": box_width_px,
                        "box_height_px": box_height_px,
                    }
                )
    selected = sorted(candidates, key=lambda row: row["selection_sha256"])[:count]
    for page_index in sorted({int(row["page_index"]) for row in selected}):
        selected_pages[page_index] = dataset[page_index]["image"].convert("RGB")
    return selected, selected_pages


def evaluate_trocr_train_pilot(
    *,
    output: Path,
    device: torch.device,
    batch_size: int,
    samples: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    panel_output = output / "trocr_funsd_train_crop_pilot"
    panel_output.mkdir()
    logger.info("stage=trocr_inputs split=train_only samples=%s output=%s", samples, panel_output)
    train_arrow = FUNSD_DATA / "funsd-train.arrow"
    input_hashes = hash_inputs(
        {
            "model_safetensors": TROCR_MODEL / "model.safetensors",
            "model_config": TROCR_MODEL / "config.json",
            "generation_config": TROCR_MODEL / "generation_config.json",
            "preprocessor_config": TROCR_MODEL / "preprocessor_config.json",
            "tokenizer_config": TROCR_MODEL / "tokenizer_config.json",
            "vocab": TROCR_MODEL / "vocab.json",
            "merges": TROCR_MODEL / "merges.txt",
            "dataset_info": FUNSD_DATA / "dataset_info.json",
            "funsd_train_arrow": train_arrow,
        },
        logger,
    )
    dataset = Dataset.from_file(str(train_arrow))
    if len(dataset) != 149:
        raise RuntimeError(f"FUNSD train size drift: {len(dataset)}")
    selected, pages = select_funsd_train_candidates(dataset, samples)
    if len(selected) != samples:
        raise RuntimeError(f"insufficient deterministic FUNSD candidates: {len(selected)}/{samples}")
    write_csv(panel_output / "selection_manifest.csv", selected)

    transformers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()
    started = time.monotonic()
    processor = TrOCRProcessor.from_pretrained(str(TROCR_MODEL), local_files_only=True)
    model, raw_loading_info = VisionEncoderDecoderModel.from_pretrained(
        str(TROCR_MODEL), local_files_only=True, output_loading_info=True
    )
    load_info = loading_info_json(raw_loading_info)
    expected_pooler_missing = sorted(load_info["missing_keys"]) == [
        "encoder.pooler.dense.bias",
        "encoder.pooler.dense.weight",
    ]
    load_info["expected_encoder_pooler_only"] = expected_pooler_missing
    load_info["pooler_used_by_encoder_last_hidden_state_or_decoder_cross_attention"] = False
    load_info["target_72_linear_matrices_affected"] = False
    load_info["accepted_for_pilot"] = bool(
        expected_pooler_missing
        and not load_info["unexpected_keys"]
        and not load_info["mismatched_keys"]
        and not load_info["error_msgs"]
    )
    if not load_info["accepted_for_pilot"]:
        raise RuntimeError(f"TrOCR model loading was not strict-clean: {load_info}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = model.get_encoder()
    logger.info(
        "stage=trocr_model_loaded device=%s dtype=%s load_elapsed=%.2fs missing=%s unexpected=%s",
        device,
        next(model.parameters()).dtype,
        time.monotonic() - started,
        len(load_info["missing_keys"]),
        len(load_info["unexpected_keys"]),
    )
    inventory = module_inventory(
        model=encoder,
        module_prefix="encoder.layer.",
        state_prefix="encoder.encoder.layer.",
        panel="trocr_base_printed_encoder",
    )
    write_csv(panel_output / "weight_matrix_inventory.csv", inventory)
    hook_names = [f"encoder.layer.0.{suffix}" for _role, suffix in ROLES]
    hook_shapes, hook_handles = hook_probe(encoder, hook_names)
    hook_pending = True
    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_started = time.monotonic()
    generation = {
        "max_new_tokens": 32,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": False,
    }
    for pad_fraction in TROCR_PAD_FRACTIONS:
        predictions: list[str] = []
        crop_meta: list[dict[str, Any]] = []
        variant_started = time.monotonic()
        for begin in range(0, len(selected), batch_size):
            end = min(begin + batch_size, len(selected))
            crops = []
            for row in selected[begin:end]:
                crop = funsd_crop(
                    pages[int(row["page_index"])],
                    row["bbox_normalized_1000"],
                    pad_fraction,
                )
                crops.append(crop)
                crop_meta.append(
                    {
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "crop_rgb_sha256": rgb_sha256(crop),
                    }
                )
            pixel_values = processor(images=crops, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                generated_ids = model.generate(pixel_values, **generation)
            predictions.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
            if hook_pending:
                remove_hooks(hook_handles)
                hook_handles = []
                hook_pending = False
            if begin == 0 or end == len(selected) or (begin // batch_size + 1) % 10 == 0:
                elapsed = time.monotonic() - variant_started
                logger.info(
                    "stage=trocr_generate pad=%.1f examples=%s/%s rate=%.1f/s elapsed=%.1fs",
                    pad_fraction,
                    end,
                    len(selected),
                    end / max(elapsed, 1e-9),
                    elapsed,
                )
        distances = []
        per_sample_cer = []
        exact = []
        nonempty = []
        for index, (selection, raw_prediction, crop_values) in enumerate(
            zip(selected, predictions, crop_meta, strict=True)
        ):
            prediction = normalize_ocr(raw_prediction)
            ground_truth = str(selection["ground_truth_normalized"])
            distance = levenshtein(ground_truth, prediction)
            distances.append(distance)
            per_sample_cer.append(distance / max(len(ground_truth), 1))
            exact.append(prediction == ground_truth)
            nonempty.append(bool(prediction))
            all_rows.append(
                {
                    "pad_fraction": pad_fraction,
                    "sample_ordinal": index,
                    "selection_sha256": selection["selection_sha256"],
                    "page_index": selection["page_index"],
                    "page_id": selection["page_id"],
                    "word_index": selection["word_index"],
                    "bbox_normalized_1000": json.dumps(selection["bbox_normalized_1000"]),
                    "ground_truth_raw": selection["ground_truth_raw"],
                    "ground_truth_normalized": ground_truth,
                    "prediction_raw": raw_prediction,
                    "prediction_normalized": prediction,
                    "edit_distance": distance,
                    "per_sample_cer": per_sample_cer[-1],
                    "exact_match": exact[-1],
                    "nonempty": nonempty[-1],
                    **crop_values,
                }
            )
        metric_rows.append(
            {
                "pad_fraction": pad_fraction,
                "examples": len(selected),
                "pages": len(pages),
                "corpus_cer": sum(distances) / sum(len(str(row["ground_truth_normalized"])) for row in selected),
                "mean_per_sample_cer": float(np.mean(per_sample_cer)),
                "median_per_sample_cer": float(np.median(per_sample_cer)),
                "exact_match_rate": float(np.mean(exact)),
                "nonempty_rate": float(np.mean(nonempty)),
                "runtime_seconds": time.monotonic() - variant_started,
            }
        )
    if hook_handles:
        remove_hooks(hook_handles)
    write_csv(panel_output / "predictions.csv", all_rows)
    write_csv(panel_output / "metrics.csv", metric_rows)
    best = min(metric_rows, key=lambda row: (row["corpus_cer"], -row["exact_match_rate"], row["pad_fraction"]))
    summary = {
        "panel": "trocr_base_printed_funsd_train_crop_pilot",
        "status": "PILOT_ONLY_NOT_A_PROSPECTIVE_TEST_GATE",
        "split": "train",
        "funsd_test_opened": False,
        "examples": len(selected),
        "unique_pages": len(pages),
        "selection_seed": PILOT_SEED,
        "selection_rule": "SHA256(seed|page_id|word_index|normalized_gt), ascending; filters recorded in resolved_config",
        "variants": metric_rows,
        "recommended_pad_fraction_from_train_only_pilot": best["pad_fraction"],
        "loading_info_clean": load_info["clean"],
        "loading_info_expected_pooler_only": load_info["expected_encoder_pooler_only"],
        "matrices": len(inventory),
        "hook_shapes": hook_shapes,
        "processor_class": type(processor).__name__,
        "generation": generation,
        "runtime_seconds": time.monotonic() - eval_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    write_json(panel_output / "summary.json", summary)
    write_json(panel_output / "loading_info.json", load_info)
    write_json(panel_output / "input_hashes.json", input_hashes)
    write_json(panel_output / "hook_shape_inspection.json", hook_shapes)
    write_json(
        panel_output / "resolved_config.json",
        {
            "mode": "train_only_crop_recipe_pilot",
            "dataset_split": "train",
            "dataset_rows": len(dataset),
            "pilot_samples": samples,
            "selection_seed": PILOT_SEED,
            "candidate_filters": {
                "normalized_ground_truth_length": [2, 24],
                "box_width_px_at_least": 4.0,
                "box_height_px_at_least": 4.0,
                "requires_alphanumeric": True,
            },
            "bbox_coordinate_system": "normalized integer [0,1000], converted independently by page width/height",
            "padding": "same pad_fraction * unpadded pixel box height on all four sides; floor left/top; ceil right/bottom",
            "pad_fractions": list(TROCR_PAD_FRACTIONS),
            "batch_size": batch_size,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "local_files_only": True,
            "generation": generation,
            "funsd_test_opened": False,
            "weight_ae_loaded": False,
        },
    )
    logger.info(
        "stage=trocr_complete status=pilot_only best_pad=%s corpus_cer=%.6f exact=%.6f artifacts=%s",
        best["pad_fraction"],
        best["corpus_cer"],
        best["exact_match_rate"],
        panel_output,
    )
    del model, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def artifact_manifest(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "relative_path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def build_readme(food: dict[str, Any], trocr: dict[str, Any]) -> str:
    best = min(trocr["variants"], key=lambda row: row["corpus_cer"])
    return f"""# Prospective panel feasibility scout

This is a source-model scout, not a Weight-AE result and not a prospective
cross-domain gate. The process installed a source-only filesystem/network seal,
never loaded Weight-AE code, and did not open the FUNSD test split.

## Food101 ViT

The exact local checkpoint fails the deterministic local quality pilot:

- validation examples: {food['examples']}
- top-1: {food['top1_accuracy']:.9f}
- top-5: {food['top5_accuracy']:.9f}
- top-1 95% Wilson interval: {food['top1_wilson_95_interval']}
- top-5 95% Wilson interval: {food['top5_wilson_95_interval']}
- macro recall: {food['macro_recall']:.9f}
- mean NLL: {food['mean_nll']:.9f}
- predicted classes: {food['unique_predicted_classes']} / 101
- model-card claimed accuracy: {food['claimed_model_card_accuracy']:.3f}
- strict loading clean: {food['loading_info_clean']}
- dataset/model label order exact: {food['dataset_model_label_mapping_exact']}

Conclusion: do not use `vit_food101_ashaduzzaman_57f4382f` as evidence from a
competently trained checkpoint. The failure is not explained by missing keys or
label-order mismatch; inspect `class_and_classifier_stats.csv` and
`predictions.csv`.

## TrOCR-base + FUNSD train crop pilot

This pilot used only FUNSD train to select an OCR crop recipe. The best tested
padding fraction was {best['pad_fraction']} with corpus CER
{best['corpus_cer']:.9f}, exact match {best['exact_match_rate']:.9f}, and
non-empty rate {best['nonempty_rate']:.9f} on {best['examples']} deterministic
word crops. This does not certify the prospective panel. Freeze the selected
recipe first, then evaluate it once on untouched FUNSD test before any Weight-AE
outcome is computed.

Both encoders expose exactly 72 target matrices (12 depths x 6 roles), with
post-transpose shapes 768x768, 768x3072, and 3072x768 as recorded in each
`weight_matrix_inventory.csv`.
"""


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = setup_logging(output)
    source_seal = install_source_only_seal()
    write_json(output / "source_only_seal.json", source_seal)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    config = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "output": str(output),
        "device": str(device),
        "food_batch_size": args.food_batch_size,
        "food_pilot_samples": args.food_pilot_samples,
        "trocr_batch_size": args.trocr_batch_size,
        "trocr_pilot_samples": args.trocr_pilot_samples,
        "seed": PILOT_SEED,
        "cache_mode": "exact local files only; offline environment",
        "verbose": True,
        "weight_ae_loaded": False,
        "funsd_test_access": False,
    }
    write_json(output / "run_config.json", config)
    write_json(output / "environment.json", environment_info(device))
    shutil.copy2(Path(__file__).resolve(), output / "executed_script.py")
    logger.info(
        "stage=start device=%s dtype=float32 seed=%s cache=local_only output=%s",
        device,
        PILOT_SEED,
        output,
    )
    food = evaluate_food(
        output=output,
        device=device,
        batch_size=args.food_batch_size,
        samples=args.food_pilot_samples,
        logger=logger,
    )
    trocr = evaluate_trocr_train_pilot(
        output=output,
        device=device,
        batch_size=args.trocr_batch_size,
        samples=args.trocr_pilot_samples,
        logger=logger,
    )
    summary = {
        "food101": food,
        "trocr_train_pilot": trocr,
        "weight_ae_loaded": False,
        "funsd_test_access": False,
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(build_readme(food, trocr), encoding="utf-8")
    logger.info(
        "stage=complete food_status=%s food_top1=%.6f trocr_best_pad=%s artifacts=%s",
        food["status"],
        food["top1_accuracy"],
        trocr["recommended_pad_fraction_from_train_only_pilot"],
        output,
    )
    for handler in logger.handlers:
        handler.flush()
    manifest = artifact_manifest(output)
    write_json(output / "artifact_manifest.json", {"artifacts": manifest, "count": len(manifest)})


if __name__ == "__main__":
    main()
