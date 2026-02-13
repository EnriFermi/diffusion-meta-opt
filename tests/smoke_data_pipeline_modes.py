from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check the full data pipeline in one of modes: none, local_disk, s3_bridge"
    )
    parser.add_argument("--mode", choices=["none", "local_disk", "s3_bridge"], required=True)

    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--datasets", default="coco2017,scene_parse_150")
    parser.add_argument(
        "--dataset-model",
        action="append",
        default=[],
        help="Dataset->models mapping in form: dataset=model_a,model_b (can repeat)",
    )

    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--collector-device", default="null")
    parser.add_argument("--collector-mode", default="auto", choices=["auto", "async", "interleaved"])

    parser.add_argument("--target-samples", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--poll-sleep", type=float, default=0.05)
    parser.add_argument("--predownload", action="store_true")
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--chunk-size-samples", type=int, default=64)
    parser.add_argument("--local-max-ready-chunks", type=int, default=40)
    parser.add_argument("--local-low-watermark-chunks", type=int, default=20)

    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default="diffusion-meta-opt/streaming")
    parser.add_argument("--s3-region", default=None)
    parser.add_argument("--s3-endpoint-url", default=None)
    parser.add_argument("--s3-max-remote-chunks", type=int, default=100)
    parser.add_argument("--delete-remote-after", choices=["consume", "download"], default="consume")

    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--weight-preview-rows", type=int, default=4)
    parser.add_argument("--weight-preview-cols", type=int, default=4)
    parser.add_argument(
        "--dump-first-weight",
        default=None,
        help="Optional path to save full first-sample weight tensor (.pt)",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return PROJECT_ROOT


def _load_cfg(overrides: list[str]) -> DictConfig:
    config_dir = _project_root() / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name="config", overrides=overrides)


def _parse_dataset_model_map(items: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for raw in items:
        value = raw.strip()
        if not value:
            continue
        if "=" not in value:
            raise ValueError(f"Invalid --dataset-model value '{raw}', expected dataset=model_a,model_b")
        dataset_name, model_csv = value.split("=", 1)
        dataset_name = dataset_name.strip()
        models = [item.strip() for item in model_csv.split(",") if item.strip()]
        if not dataset_name or not models:
            raise ValueError(f"Invalid --dataset-model value '{raw}'")
        mapping[dataset_name] = models
    return mapping


def _build_overrides(args: argparse.Namespace) -> list[str]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    if not datasets:
        raise ValueError("--datasets is empty")

    overrides: list[str] = [
        f"data.path={args.data_root}",
        f"data.enabled_datasets=[{','.join(datasets)}]",
        f"collector.mode={args.collector_mode}",
        f"collector.device={args.collector_device}",
        f"train.device={args.train_device}",
        f"streaming.mode={args.mode}",
        f"streaming.chunk_size_samples={int(args.chunk_size_samples)}",
    ]

    if args.hf_token:
        overrides.append(f"hf.token={args.hf_token}")

    mapping = _parse_dataset_model_map(args.dataset_model)
    if not mapping:
        mapping = {
            "coco2017": ["clip_vit_b32"],
            "scene_parse_150": ["clip_vit_b32"],
            "cc12m": ["clip_vit_b32"],
        }

    for dataset_name in datasets:
        models = mapping.get(dataset_name)
        if models:
            overrides.append(f"+data.dataset_overrides.{dataset_name}.models=[{','.join(models)}]")

    if args.mode == "local_disk":
        local_root = Path(args.data_root) / "streaming" / "local_disk_smoke"
        producer_dir = Path(args.data_root) / "streaming" / "spool" / "producer_smoke"
        consumer_dir = Path(args.data_root) / "streaming" / "cache" / "consumer_smoke"

        overrides.extend(
            [
                f"streaming.local_disk.root_dir={local_root}",
                f"streaming.local_disk.max_ready_chunks={int(args.local_max_ready_chunks)}",
                f"streaming.local_disk.low_watermark_chunks={int(args.local_low_watermark_chunks)}",
                f"streaming.producer.local_spool_dir={producer_dir}",
                f"streaming.consumer.local_cache_dir={consumer_dir}",
                "streaming.distributed.enabled=false",
            ]
        )

    if args.mode == "s3_bridge":
        if not args.s3_bucket:
            raise ValueError("--s3-bucket is required for --mode s3_bridge")

        producer_dir = Path(args.data_root) / "streaming" / "spool" / "producer_s3_smoke"
        consumer_dir = Path(args.data_root) / "streaming" / "cache" / "consumer_s3_smoke"

        overrides.extend(
            [
                f"streaming.s3.bucket={args.s3_bucket}",
                f"streaming.s3.prefix={args.s3_prefix}",
                f"streaming.s3.max_remote_chunks={int(args.s3_max_remote_chunks)}",
                f"streaming.consumer.delete_remote_after={args.delete_remote_after}",
                f"streaming.producer.local_spool_dir={producer_dir}",
                f"streaming.consumer.local_cache_dir={consumer_dir}",
            ]
        )
        if args.s3_region:
            overrides.append(f"streaming.s3.region={args.s3_region}")
        if args.s3_endpoint_url:
            overrides.append(f"streaming.s3.endpoint_url={args.s3_endpoint_url}")

    return overrides


def _weight_preview(weight: torch.Tensor, rows: int, cols: int) -> list[list[float]]:
    if weight.ndim != 2:
        flat = weight.reshape(-1)
        take = max(1, min(int(rows * cols), int(flat.numel())))
        return [flat[:take].detach().to("cpu", dtype=torch.float32).tolist()]

    r = max(1, min(int(rows), int(weight.shape[0])))
    c = max(1, min(int(cols), int(weight.shape[1])))
    chunk = weight[:r, :c].detach().to("cpu", dtype=torch.float32)
    return chunk.tolist()


def _run_smoke(
    cfg: DictConfig,
    target_samples: int,
    timeout_seconds: int,
    poll_sleep: float,
    predownload: bool,
    weight_preview_rows: int,
    weight_preview_cols: int,
    dump_first_weight: str | None,
) -> dict[str, Any]:
    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    model_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()

    consumed = 0
    first_sample_summary: dict[str, Any] | None = None
    started = time.time()
    step = 0

    try:
        if predownload:
            collector.predownload_models()

        collector.start()

        while consumed < target_samples:
            if time.time() - started > timeout_seconds:
                raise TimeoutError(
                    f"Timeout: collected only {consumed}/{target_samples} samples in {timeout_seconds}s"
                )

            if not collector.is_async_mode:
                dataset.maybe_collect(step)

            sample = dataset.try_next_sample()
            if sample is None:
                time.sleep(poll_sleep)
                step += 1
                continue

            if not torch.is_tensor(sample.x) or not torch.is_tensor(sample.y):
                raise TypeError("SharedSample x/y must be tensors")
            if not torch.is_tensor(sample.weight):
                raise TypeError("SharedSample weight must be tensor")
            if sample.x.dtype != torch.float32 or sample.y.dtype != torch.float32:
                raise TypeError(f"Expected float32 tensors, got x={sample.x.dtype}, y={sample.y.dtype}")
            if str(sample.x.device) != "cpu" or str(sample.y.device) != "cpu":
                raise TypeError(f"Expected CPU tensors, got x={sample.x.device}, y={sample.y.device}")

            if first_sample_summary is None:
                weight = sample.weight.detach().to("cpu", dtype=torch.float32)

                weight_dump_path = None
                if dump_first_weight:
                    dump_path = Path(dump_first_weight)
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(weight, dump_path)
                    weight_dump_path = str(dump_path.resolve())

                first_sample_summary = {
                    "model_name": sample.model_name,
                    "layer_name": sample.layer_name,
                    "x_shape": tuple(sample.x.shape),
                    "y_shape": tuple(sample.y.shape),
                    "weight_shape": tuple(weight.shape),
                    "weight_dtype": str(weight.dtype),
                    "weight_device": str(weight.device),
                    "weight_l2_norm": float(weight.norm().item()),
                    "weight_preview": _weight_preview(
                        weight=weight,
                        rows=weight_preview_rows,
                        cols=weight_preview_cols,
                    ),
                    "weight_dump_path": weight_dump_path,
                    "meta_keys": sorted(sample.meta.keys()),
                }

            model_counts[sample.model_name] += 1
            layer_counts[sample.layer_name] += 1
            for item in sample.meta.get("image_meta", []):
                ds_name = item.get("dataset_name")
                if ds_name:
                    dataset_counts[str(ds_name)] += 1

            consumed += 1
            step += 1

    finally:
        dataset.close()
        collector.shutdown()

    if consumed <= 0:
        raise RuntimeError("Smoke run produced zero samples")

    return {
        "ok": True,
        "consumed": consumed,
        "target_samples": target_samples,
        "elapsed_seconds": round(time.time() - started, 3),
        "collector_mode": collector.collector_mode,
        "streaming_mode": collector.streaming_mode,
        "cache_metric_after": collector.cache_size(),
        "model_counts": dict(model_counts),
        "dataset_counts": dict(dataset_counts),
        "top_layers": layer_counts.most_common(10),
        "first_sample": first_sample_summary,
        "collector_stats": collector.stats(),
    }


def main() -> int:
    args = _parse_args()
    overrides = _build_overrides(args)
    cfg = _load_cfg(overrides)

    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg, resolve=True))

    try:
        result = _run_smoke(
            cfg=cfg,
            target_samples=int(args.target_samples),
            timeout_seconds=int(args.timeout_seconds),
            poll_sleep=float(args.poll_sleep),
            predownload=bool(args.predownload),
            weight_preview_rows=int(args.weight_preview_rows),
            weight_preview_cols=int(args.weight_preview_cols),
            dump_first_weight=args.dump_first_weight,
        )
    except Exception as exc:
        message = str(exc)
        hint = None
        if "Can't load image processor" in message and not args.predownload:
            hint = (
                "Model artifacts are likely missing in local cache while runtime uses local_files_only=True. "
                "Retry with --predownload or keep --predownload in smoke_data_pipeline_modes.sh."
            )
        payload: dict[str, Any] = {"ok": False, "error": message}
        if hint:
            payload["hint"] = hint
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
