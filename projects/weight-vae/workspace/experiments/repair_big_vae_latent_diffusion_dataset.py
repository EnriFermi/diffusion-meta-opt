from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from big_vae.datasets.latent_diffusion import (
    OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
    _directory_size_bytes,
    _prepare_cpu_tensor,
)
from big_vae.datasets.offline import infer_layer_depth, infer_layer_type


LOGGER = logging.getLogger("repair_big_vae_latent_diffusion_dataset")


def _pow2_bucket(value: int) -> str:
    value = int(value)
    if value <= 0:
        return "0"
    lower = 1 << int(math.floor(math.log2(value)))
    upper = max(lower, (lower << 1) - 1)
    return f"{lower}-{upper}"


def _shape_key(d_in: int, d_out: int) -> str:
    return f"{int(d_in)}x{int(d_out)}"


def _sample_dataset_names(meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(meta, dict):
        return []
    image_meta = meta.get("image_meta", [])
    if not isinstance(image_meta, list):
        return []
    dataset_names = {
        str(item.get("dataset_name")).strip()
        for item in image_meta
        if isinstance(item, dict) and str(item.get("dataset_name", "")).strip()
    }
    return sorted(dataset_names)


def _primary_dataset_name(meta: dict[str, Any] | None) -> str:
    dataset_names = _sample_dataset_names(meta)
    return dataset_names[0] if dataset_names else "<unknown_dataset>"


def _counter_gini(counter: Counter[str]) -> float:
    counts = sorted(int(value) for value in counter.values() if int(value) > 0)
    if not counts:
        return 0.0
    n = len(counts)
    total = float(sum(counts))
    weighted = sum((idx + 1) * value for idx, value in enumerate(counts))
    return float((2.0 * weighted) / (n * total) - (n + 1) / n)


def _counter_report(counter: Counter[str], *, topk: int, item_key: str) -> dict[str, Any]:
    total = int(sum(counter.values()))
    unique = int(len(counter))
    if total <= 0 or unique <= 0:
        return {
            "total": 0,
            "unique": 0,
            "top1_share": 0.0,
            "top5_share": 0.0,
            "top10_share": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 1.0,
            "perplexity": 0.0,
            "gini": 0.0,
            "items": [],
        }

    probs = [float(count) / float(total) for _, count in counter.items() if int(count) > 0]
    entropy = float(-sum(p * math.log(p) for p in probs if p > 0.0))
    max_entropy = float(math.log(unique)) if unique > 1 else 0.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0.0 else 1.0
    top_items = counter.most_common(max(1, int(topk)))
    return {
        "total": total,
        "unique": unique,
        "top1_share": float(top_items[0][1]) / float(total),
        "top5_share": float(sum(count for _, count in counter.most_common(5))) / float(total),
        "top10_share": float(sum(count for _, count in counter.most_common(10))) / float(total),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "perplexity": float(math.exp(entropy)),
        "gini": _counter_gini(counter),
        "items": [
            {
                item_key: key,
                "count": int(count),
                "share": float(count) / float(total),
            }
            for key, count in top_items
        ],
    }


def _top_items(counter: Counter[str], *, topk: int, item_key: str) -> list[dict[str, Any]]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return []
    return [
        {
            item_key: key,
            "count": int(count),
            "share": float(count) / total,
        }
        for key, count in counter.most_common(max(1, int(topk)))
    ]


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload)!r}")
    return payload


def _record_decoder_aux_complete(record: dict[str, Any]) -> bool:
    required = ("X", "W", "x_mask", "d_in_mask", "d_out_mask")
    return all(torch.is_tensor(record.get(key)) for key in required)


def _infer_slice_shape(
    record: dict[str, Any],
    *,
    existing_manifest: dict[str, Any] | None,
) -> tuple[int | None, int | None, int | None]:
    slice_shape = (existing_manifest or {}).get("slice_shape", {})
    if not isinstance(slice_shape, dict):
        slice_shape = {}
    target_T = int(record.get("target_T_patches", 0) or 0)
    target_d_out = int(record.get("target_d_out", 0) or 0)
    meta = record.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    if target_T <= 0:
        target_T = int(meta.get("latent_diffusion_target_T_patches", 0) or 0)
    target_d_in = int(meta.get("latent_diffusion_target_d_in", 0) or 0)
    if target_T <= 0:
        target_T = int(slice_shape.get("target_T_patches", 0) or 0)
    if target_d_out <= 0:
        target_d_out = int(slice_shape.get("target_d_out", 0) or 0)
    patch_size = int(slice_shape.get("patch_size", 0) or 0)
    if patch_size <= 0 and target_T > 0 and target_d_in > 0 and target_d_in % target_T == 0:
        patch_size = target_d_in // target_T
    if patch_size <= 0:
        patch_size = None
    if target_T <= 0:
        target_T = None
    if target_d_out <= 0:
        target_d_out = None
    return patch_size, target_T, target_d_out


def _validate_record(record: dict[str, Any], *, chunk_path: Path, record_idx: int) -> tuple[int, int]:
    latent_mu = record.get("latent_mu")
    cond_patch = record.get("cond_patch")
    patch_mask = record.get("patch_mask")
    if not torch.is_tensor(latent_mu) or latent_mu.ndim != 1:
        raise TypeError(f"Invalid latent_mu in {chunk_path} record_idx={record_idx}")
    if not torch.is_tensor(cond_patch) or cond_patch.ndim != 2:
        raise TypeError(f"Invalid cond_patch in {chunk_path} record_idx={record_idx}")
    if not torch.is_tensor(patch_mask) or patch_mask.ndim != 1:
        raise TypeError(f"Invalid patch_mask in {chunk_path} record_idx={record_idx}")
    if int(cond_patch.shape[0]) != int(patch_mask.shape[0]):
        raise ValueError(
            f"cond_patch/patch_mask mismatch in {chunk_path} record_idx={record_idx}: "
            f"{tuple(cond_patch.shape)} vs {tuple(patch_mask.shape)}"
        )
    return int(latent_mu.numel()), int(cond_patch.shape[1])


def _iter_chunk_records(root: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    chunks_dir = root / "chunks"
    chunk_paths = sorted(chunks_dir.glob("*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No latent diffusion chunks found under {chunks_dir}")
    for chunk_path in chunk_paths:
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Chunk payload must be a dict: {chunk_path}")
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise TypeError(f"Chunk records payload must be a list: {chunk_path}")
        for record_idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"Record must be a dict in {chunk_path} record_idx={record_idx}")
            yield chunk_path, record_idx, record


def _empty_counter_bundle() -> dict[str, Counter[str]]:
    return {
        "datasets": Counter(),
        "models": Counter(),
        "layer_types": Counter(),
        "depths": Counter(),
        "shapes": Counter(),
        "d_in_buckets": Counter(),
        "d_out_buckets": Counter(),
        "num_params_buckets": Counter(),
        "sources": Counter(),
    }


def _accumulate_record_counters(counters: dict[str, Counter[str]], record: dict[str, Any]) -> None:
    model_name = str(record.get("model_name", "") or "").strip() or "<unknown_model>"
    layer_name = str(record.get("layer_name", "") or "").strip() or "<unknown_layer>"
    meta = record.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    dataset_name = _primary_dataset_name(meta)
    d_in = int(record.get("d_in", 0) or 0)
    d_out = int(record.get("d_out", 0) or 0)
    layer_type = str(record.get("layer_type", "") or "").strip() or infer_layer_type(layer_name)
    depth_value = record.get("layer_depth", infer_layer_depth(layer_name))
    depth_label = str(depth_value) if depth_value is not None else "unknown"
    counters["datasets"][dataset_name] += 1
    counters["models"][model_name] += 1
    counters["layer_types"][layer_type] += 1
    counters["depths"][depth_label] += 1
    counters["shapes"][_shape_key(d_in, d_out)] += 1
    counters["d_in_buckets"][_pow2_bucket(d_in)] += 1
    counters["d_out_buckets"][_pow2_bucket(d_out)] += 1
    counters["num_params_buckets"][_pow2_bucket(d_in * d_out)] += 1
    source_key = str(record.get("source_key", "") or "").strip() or f"{model_name}:{layer_name}"
    counters["sources"][source_key] += 1


def _balance_report_from_counters(counters: dict[str, Counter[str]], *, topk: int) -> dict[str, Any]:
    return {
        "datasets": _counter_report(counters["datasets"], topk=topk, item_key="dataset_name"),
        "models": _counter_report(counters["models"], topk=topk, item_key="model_name"),
        "layer_types": _counter_report(counters["layer_types"], topk=topk, item_key="layer_type"),
        "depths": _counter_report(counters["depths"], topk=topk, item_key="depth"),
        "shapes": _counter_report(counters["shapes"], topk=topk, item_key="shape"),
        "d_in_buckets": _counter_report(counters["d_in_buckets"], topk=topk, item_key="d_in_bucket"),
        "d_out_buckets": _counter_report(counters["d_out_buckets"], topk=topk, item_key="d_out_bucket"),
        "num_params_buckets": _counter_report(
            counters["num_params_buckets"],
            topk=topk,
            item_key="num_params_bucket",
        ),
        "sources": _counter_report(counters["sources"], topk=topk, item_key="source_key"),
    }


def _prefix_thresholds(total_records: int, fractions: Sequence[float]) -> list[tuple[float, int]]:
    if total_records <= 0:
        return []
    thresholds: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in fractions:
        if float(fraction) <= 0.0:
            continue
        count = min(total_records, max(1, int(math.ceil(float(fraction) * float(total_records)))))
        if count in seen:
            continue
        thresholds.append((float(fraction), count))
        seen.add(count)
    if total_records not in seen:
        thresholds.append((1.0, total_records))
    thresholds.sort(key=lambda item: item[1])
    return thresholds


def _skew_hints(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    model_top1 = float(report["models"].get("top1_share", 0.0))
    shape_top1 = float(report["shapes"].get("top1_share", 0.0))
    dataset_top1 = float(report["datasets"].get("top1_share", 0.0))
    if model_top1 >= 0.5:
        hints.append(f"record distribution is model-heavy: top model share={model_top1:.1%}")
    if shape_top1 >= 0.35:
        hints.append(f"record distribution is shape-heavy: top exact shape share={shape_top1:.1%}")
    if dataset_top1 >= 0.5:
        hints.append(f"record distribution is dataset-heavy: top dataset share={dataset_top1:.1%}")
    return hints


def repair_big_vae_latent_diffusion_dataset_root(
    root_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    topk: int = 20,
    prefix_fractions: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    write_manifest: bool = True,
    write_stats: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    root = Path(str(root_dir)).expanduser().resolve()
    logger_local = logger or LOGGER
    existing_manifest = _optional_json(root / "manifest.json")

    total_records = 0
    latent_sum: torch.Tensor | None = None
    latent_sumsq: torch.Tensor | None = None
    z_dim: int | None = None
    cond_dim: int | None = None
    num_chunks = len(sorted((root / "chunks").glob("*.pt")))
    unique_sources: set[str] = set()
    unique_models: set[str] = set()
    unique_layers: set[str] = set()
    decoder_aux_records = 0
    missing_decoder_aux_records = 0
    patch_size_candidates: set[int] = set()
    target_T_candidates: set[int] = set()
    target_d_out_candidates: set[int] = set()
    overall_counters = _empty_counter_bundle()

    for chunk_path, record_idx, record in _iter_chunk_records(root):
        record_z_dim, record_cond_dim = _validate_record(record, chunk_path=chunk_path, record_idx=record_idx)
        if z_dim is None:
            z_dim = record_z_dim
            latent_sum = torch.zeros(z_dim, dtype=torch.float64)
            latent_sumsq = torch.zeros(z_dim, dtype=torch.float64)
        if cond_dim is None:
            cond_dim = record_cond_dim
        if record_z_dim != int(z_dim):
            raise ValueError(f"Inconsistent z_dim in {chunk_path} record_idx={record_idx}: {record_z_dim} vs {z_dim}")
        if record_cond_dim != int(cond_dim):
            raise ValueError(
                f"Inconsistent cond_dim in {chunk_path} record_idx={record_idx}: {record_cond_dim} vs {cond_dim}"
            )
        assert latent_sum is not None
        assert latent_sumsq is not None
        latent_mu = _prepare_cpu_tensor(record["latent_mu"]).to(dtype=torch.float64)
        latent_sum += latent_mu
        latent_sumsq += latent_mu.pow(2)
        total_records += 1

        model_name = str(record.get("model_name", "") or "").strip() or "<unknown_model>"
        layer_name = str(record.get("layer_name", "") or "").strip() or "<unknown_layer>"
        source_key = str(record.get("source_key", "") or "").strip() or f"{model_name}:{layer_name}"
        unique_models.add(model_name)
        unique_layers.add(layer_name)
        unique_sources.add(source_key)
        _accumulate_record_counters(overall_counters, record)

        if _record_decoder_aux_complete(record):
            decoder_aux_records += 1
        else:
            missing_decoder_aux_records += 1

        patch_size, target_T, target_d_out = _infer_slice_shape(record, existing_manifest=existing_manifest)
        if patch_size is not None:
            patch_size_candidates.add(int(patch_size))
        if target_T is not None:
            target_T_candidates.add(int(target_T))
        if target_d_out is not None:
            target_d_out_candidates.add(int(target_d_out))

    if total_records <= 0 or z_dim is None or cond_dim is None or latent_sum is None or latent_sumsq is None:
        raise RuntimeError(f"No valid records found under {root / 'chunks'}")

    latent_mean = latent_sum / float(total_records)
    latent_var = (latent_sumsq / float(total_records)) - latent_mean.pow(2)
    latent_std = torch.sqrt(latent_var.clamp_min(1e-6))
    stats_payload = {
        "latent_mean": latent_mean.to(dtype=torch.float32),
        "latent_std": latent_std.to(dtype=torch.float32),
        "count": int(total_records),
        "z_dim": int(z_dim),
        "cond_dim": int(cond_dim),
    }

    manifest_slice_shape = dict((existing_manifest or {}).get("slice_shape", {}) or {})
    repaired_slice_shape = {
        "patch_size": int(next(iter(patch_size_candidates))) if len(patch_size_candidates) == 1 else int(manifest_slice_shape.get("patch_size", 0) or 0),
        "target_T_patches": int(next(iter(target_T_candidates))) if len(target_T_candidates) == 1 else int(manifest_slice_shape.get("target_T_patches", 0) or 0),
        "target_d_in": 0,
        "target_d_out": int(next(iter(target_d_out_candidates))) if len(target_d_out_candidates) == 1 else int(manifest_slice_shape.get("target_d_out", 0) or 0),
    }
    if repaired_slice_shape["patch_size"] > 0 and repaired_slice_shape["target_T_patches"] > 0:
        repaired_slice_shape["target_d_in"] = (
            int(repaired_slice_shape["patch_size"]) * int(repaired_slice_shape["target_T_patches"])
        )

    manifest_payload = {
        "format_version": OFFLINE_BIG_VAE_LATENT_DIFFUSION_FORMAT_VERSION,
        "created_at": float((existing_manifest or {}).get("created_at", 0.0) or 0.0),
        "repaired_at": float(time.time()),
        "root_dir": str(root),
        "accepted_records": int(total_records),
        "num_chunks": int(num_chunks),
        "z_dim": int(z_dim),
        "cond_dim": int(cond_dim),
        "actual_size_bytes": int(_directory_size_bytes(root)),
        "actual_size_gb": float(_directory_size_bytes(root)) / (1024.0 ** 3),
        "has_decoder_aux_tensors": bool(decoder_aux_records == total_records),
        "slice_shape": repaired_slice_shape,
        "layout": {
            "chunks_dir": "chunks",
            "stats_path": "latent_stats.pt",
        },
        "config_snapshot": dict((existing_manifest or {}).get("config_snapshot", {}) or {}),
        "repair_metadata": {
            "existing_manifest_found": bool(existing_manifest is not None),
            "decoder_aux_records": int(decoder_aux_records),
            "missing_decoder_aux_records": int(missing_decoder_aux_records),
            "patch_size_candidates": sorted(int(value) for value in patch_size_candidates),
            "target_T_patches_candidates": sorted(int(value) for value in target_T_candidates),
            "target_d_out_candidates": sorted(int(value) for value in target_d_out_candidates),
        },
    }

    if write_stats:
        torch.save(stats_payload, root / "latent_stats.pt")
    if write_manifest:
        (root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    prefix_reports: list[dict[str, Any]] = []
    thresholds = _prefix_thresholds(total_records, prefix_fractions)
    if thresholds:
        threshold_idx = 0
        prefix_counters = _empty_counter_bundle()
        seen_records = 0
        for _chunk_path, _record_idx, record in _iter_chunk_records(root):
            seen_records += 1
            _accumulate_record_counters(prefix_counters, record)
            while threshold_idx < len(thresholds) and seen_records >= thresholds[threshold_idx][1]:
                fraction, count = thresholds[threshold_idx]
                prefix_report = _balance_report_from_counters(prefix_counters, topk=topk)
                prefix_report["fraction"] = float(fraction)
                prefix_report["num_records"] = int(count)
                prefix_report["skew_hints"] = _skew_hints(prefix_report)
                prefix_reports.append(prefix_report)
                threshold_idx += 1

    overall_report = _balance_report_from_counters(overall_counters, topk=topk)
    overall_report["skew_hints"] = _skew_hints(overall_report)
    report = {
        "root_dir": str(root),
        "manifest_path": str(root / "manifest.json"),
        "stats_path": str(root / "latent_stats.pt"),
        "accepted_records": int(total_records),
        "num_chunks": int(num_chunks),
        "z_dim": int(z_dim),
        "cond_dim": int(cond_dim),
        "actual_size_bytes": int(_directory_size_bytes(root)),
        "actual_size_gb": float(_directory_size_bytes(root)) / (1024.0 ** 3),
        "unique_sources": int(len(unique_sources)),
        "unique_models": int(len(unique_models)),
        "unique_layers": int(len(unique_layers)),
        "slice_shape": repaired_slice_shape,
        "has_decoder_aux_tensors": bool(decoder_aux_records == total_records),
        "decoder_aux_tensor_records": int(decoder_aux_records),
        "missing_decoder_aux_tensor_records": int(missing_decoder_aux_records),
        "overall": overall_report,
        "prefix_balance": prefix_reports,
        "repair_warnings": [],
    }
    if len(patch_size_candidates) > 1:
        report["repair_warnings"].append(f"inconsistent patch_size candidates: {sorted(patch_size_candidates)}")
    if len(target_T_candidates) > 1:
        report["repair_warnings"].append(f"inconsistent target_T_patches candidates: {sorted(target_T_candidates)}")
    if len(target_d_out_candidates) > 1:
        report["repair_warnings"].append(f"inconsistent target_d_out candidates: {sorted(target_d_out_candidates)}")
    if decoder_aux_records not in {0, total_records}:
        report["repair_warnings"].append(
            "decoder aux tensors are present only for a subset of records; dataset is mixed-format"
        )

    output_root = Path(output_dir).expanduser().resolve() if output_dir is not None else (root / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "latent_diffusion_repair_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger_local.info(
        "Latent diffusion dataset repair complete: root=%s records=%s chunks=%s report=%s",
        root,
        int(total_records),
        int(num_chunks),
        report_path,
    )
    return report


def _summary_lines(report: dict[str, Any]) -> list[str]:
    overall = report.get("overall", {})
    models = overall.get("models", {})
    shapes = overall.get("shapes", {})
    datasets = overall.get("datasets", {})
    return [
        (
            "Latent diffusion dataset repair summary: "
            f"records={int(report.get('accepted_records', 0))} "
            f"chunks={int(report.get('num_chunks', 0))} "
            f"size_gb={float(report.get('actual_size_gb', 0.0)):.2f} "
            f"models={int(report.get('unique_models', 0))} "
            f"sources={int(report.get('unique_sources', 0))}"
        ),
        (
            "Overall balance: "
            f"model_top1={float(models.get('top1_share', 0.0)):.1%} "
            f"shape_top1={float(shapes.get('top1_share', 0.0)):.1%} "
            f"dataset_top1={float(datasets.get('top1_share', 0.0)):.1%}"
        ),
        f"Report path: {report.get('report_path', '')}",
    ]


def _parse_prefix_fractions(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return tuple(values or (0.01, 0.05, 0.1, 0.25, 0.5, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair interrupted BigVAE latent diffusion dataset and analyze balance.")
    parser.add_argument("root_dir", type=str, help="Root directory of the latent diffusion dataset.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for the repair report JSON.")
    parser.add_argument("--topk", type=int, default=20, help="Top-k items to include per balance section.")
    parser.add_argument(
        "--prefix-fractions",
        type=str,
        default="0.01,0.05,0.1,0.25,0.5,1.0",
        help="Comma-separated prefix fractions for balance diagnostics.",
    )
    parser.add_argument("--no-write-manifest", action="store_true", help="Do not write repaired manifest.json.")
    parser.add_argument("--no-write-stats", action="store_true", help="Do not write repaired latent_stats.pt.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = repair_big_vae_latent_diffusion_dataset_root(
        args.root_dir,
        output_dir=args.output_dir or None,
        topk=max(1, int(args.topk)),
        prefix_fractions=_parse_prefix_fractions(args.prefix_fractions),
        write_manifest=not bool(args.no_write_manifest),
        write_stats=not bool(args.no_write_stats),
        logger=LOGGER,
    )
    for line in _summary_lines(report):
        LOGGER.info(line)


if __name__ == "__main__":
    main()
