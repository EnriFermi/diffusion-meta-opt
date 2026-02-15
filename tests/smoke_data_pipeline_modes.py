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
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset
from dataset.shared.streaming.factory import build_chunk_store, resolve_streaming_cfg

DEFAULT_FULL_DATASET_PROFILE = "all_datasets_no_flickr30k"
DEFAULT_FULL_DATASET_LIST = [
    "bdd100k",
    "cc12m",
    "coco2017",
    "cord_v2",
    "doclaynet_v11",
    "docvqa_1200",
    "eurosat_rgb",
    "funsd",
    "mapillary_vistas_v2",
    "oxford_pets",
    "patchcamelyon",
    "relaion400m",
    "scene_parse_150",
    "stanford_cars",
    "visual_genome",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check the full data pipeline in one of modes: none, local_disk, s3_bridge"
    )
    parser.add_argument("--mode", choices=["none", "local_disk", "s3_bridge"], required=True)
    parser.add_argument(
        "--data-profile",
        default=DEFAULT_FULL_DATASET_PROFILE,
        help="Hydra data profile from conf/data/<profile>.yaml (default: streaming-safe set without flickr30k)",
    )

    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset subset override. If empty, profile datasets are used.",
    )
    parser.add_argument(
        "--dataset-model",
        action="append",
        default=[],
        help="Dataset->models mapping in form: dataset=model_a,model_b (can repeat)",
    )

    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--collector-device", default="cuda:1")
    parser.add_argument("--collector-mode", default="auto", choices=["auto", "async", "interleaved"])

    parser.add_argument("--target-samples", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-sleep", type=float, default=0.05)
    parser.add_argument("--predownload", action="store_true")
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--chunk-size-samples", type=int, default=16)
    parser.add_argument("--raw-chunk-size-images", type=int, default=32)
    parser.add_argument("--raw-num-chunks-kept", type=int, default=2)
    parser.add_argument("--xy-samples-random-slice", type=int, default=64)
    parser.add_argument("--local-ready-store-max-chunks", type=int, default=40)
    parser.add_argument("--local-refill-after-consumed-chunks", type=int, default=20)

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
        "--run-tag",
        default=None,
        help="Optional run tag for isolated local/S3 smoke paths. Defaults to timestamp.",
    )
    parser.add_argument("--turnover-probe-samples", type=int, default=24)
    parser.add_argument("--turnover-probe-timeout-seconds", type=int, default=180)
    parser.add_argument("--skip-turnover-probe", action="store_true")
    parser.add_argument("--diversity-window", type=int, default=128)
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
        cfg = compose(config_name="config", overrides=overrides)

    # This script composes config outside @hydra.main.
    # Replace hydra-dependent log interpolations with concrete values.
    with open_dict(cfg):
        if "logging" in cfg:
            project = str(cfg.logging.get("project_name", "diffusion_meta_opt"))
            ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            cfg.logging.file_name = f"{project}_smoke_data_pipeline_modes_{ts}.log"
            cfg.logging.file_path = f"{cfg.logging.dir}/{cfg.logging.file_name}"

    return cfg


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
    run_tag = str(args.run_tag).strip() if args.run_tag else f"run_{int(time.time())}"
    cli_datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    datasets = cli_datasets if cli_datasets else DEFAULT_FULL_DATASET_LIST

    overrides: list[str] = [
        f"data={args.data_profile}",
        f"data.path={args.data_root}",
        f"collector.mode={args.collector_mode}",
        f"collector.device={args.collector_device}",
        f"train.device={args.train_device}",
        f"streaming.mode={args.mode}",
        f"streaming.chunk_size_samples={int(args.chunk_size_samples)}",
        f"collector.layer_output_splitting.xy_samples_random_slice={int(args.xy_samples_random_slice)}",
    ]
    if datasets:
        overrides.append(f"data.enabled_datasets=[{','.join(datasets)}]")

    if args.hf_token:
        overrides.append(f"hf.token={args.hf_token}")

    mapping = _parse_dataset_model_map(args.dataset_model)

    for dataset_name in datasets:
        models = mapping.get(dataset_name)
        if models:
            overrides.append(f"+data.dataset_overrides.{dataset_name}.models=[{','.join(models)}]")
        overrides.append(
            f"+data.dataset_overrides.{dataset_name}.cache.chunk_size_images={int(args.raw_chunk_size_images)}"
        )
        overrides.append(
            f"+data.dataset_overrides.{dataset_name}.cache.num_chunks_kept={int(args.raw_num_chunks_kept)}"
        )

    if args.mode == "local_disk":
        local_root = Path(args.data_root) / "streaming" / "local_disk_smoke" / run_tag
        producer_dir = Path(args.data_root) / "streaming" / "spool" / "producer_smoke" / run_tag
        consumer_dir = Path(args.data_root) / "streaming" / "cache" / "consumer_smoke" / run_tag
        refill_after = max(1, int(args.local_refill_after_consumed_chunks))

        overrides.extend(
            [
                f"streaming.producer.ready_store_dir={local_root}",
                f"streaming.producer.ready_store_max_chunks={int(args.local_ready_store_max_chunks)}",
                f"streaming.producer.refill_after_consumed_chunks={int(refill_after)}",
                f"streaming.producer.spool_dir={producer_dir}",
                f"streaming.consumer.cache_dir={consumer_dir}",
                "streaming.distributed.enabled=false",
            ]
        )

    if args.mode == "s3_bridge":
        if not args.s3_bucket:
            raise ValueError("--s3-bucket is required for --mode s3_bridge")

        producer_dir = Path(args.data_root) / "streaming" / "spool" / "producer_s3_smoke" / run_tag
        consumer_dir = Path(args.data_root) / "streaming" / "cache" / "consumer_s3_smoke" / run_tag
        prefix = str(args.s3_prefix).strip().rstrip("/")
        if prefix:
            prefix = f"{prefix}/{run_tag}"
        else:
            prefix = run_tag

        overrides.extend(
            [
                f"streaming.s3.bucket={args.s3_bucket}",
                f"streaming.s3.prefix={prefix}",
                f"streaming.s3.max_remote_chunks={int(args.s3_max_remote_chunks)}",
                f"streaming.consumer.delete_remote_after={args.delete_remote_after}",
                f"streaming.producer.spool_dir={producer_dir}",
                f"streaming.consumer.cache_dir={consumer_dir}",
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


def _streaming_chunk_snapshot(cfg: DictConfig) -> dict[str, Any] | None:
    mode = str(cfg.streaming.mode).lower()
    if mode == "none":
        return None

    streaming_cfg = resolve_streaming_cfg(OmegaConf.to_container(cfg.streaming, resolve=True))
    store = build_chunk_store(streaming_cfg)
    if store is None:
        return None

    refs = store.list_ready(limit=2000)
    ids = [item.chunk_id for item in refs]
    return {
        "ready_count": len(ids),
        "ready_chunk_ids": ids,
    }


def _diversity_report(samples: list[Any], window_size: int) -> dict[str, Any]:
    if not samples:
        return {
            "window_size": int(window_size),
            "num_samples": 0,
            "min_unique_model_run_ids_in_window": 0,
            "min_unique_datasets_in_window": 0,
            "global_unique_model_run_ids": 0,
            "global_unique_datasets": 0,
            "max_model_run_streak": 0,
            "max_primary_dataset_streak": 0,
        }

    run_ids: list[int | None] = []
    dataset_sets: list[set[str]] = []
    primary_datasets: list[str | None] = []

    for sample in samples:
        run_value = sample.meta.get("model_run_id")
        run_ids.append(int(run_value) if run_value is not None else None)

        image_meta = sample.meta.get("image_meta", [])
        ds_set = {
            str(item.get("dataset_name"))
            for item in image_meta
            if isinstance(item, dict) and item.get("dataset_name") is not None
        }
        dataset_sets.append(ds_set)
        primary_datasets.append(next(iter(ds_set)) if ds_set else None)

    real_window = max(1, min(int(window_size), len(samples)))
    min_unique_run_ids = 10**9
    min_unique_datasets = 10**9

    for start in range(0, len(samples) - real_window + 1):
        end = start + real_window
        window_run_ids = {value for value in run_ids[start:end] if value is not None}
        window_datasets: set[str] = set()
        for ds_set in dataset_sets[start:end]:
            window_datasets.update(ds_set)
        min_unique_run_ids = min(min_unique_run_ids, len(window_run_ids))
        min_unique_datasets = min(min_unique_datasets, len(window_datasets))

    max_model_streak = 1
    max_dataset_streak = 1
    current_model_streak = 1
    current_dataset_streak = 1

    for idx in range(1, len(samples)):
        if run_ids[idx] is not None and run_ids[idx] == run_ids[idx - 1]:
            current_model_streak += 1
        else:
            current_model_streak = 1
        max_model_streak = max(max_model_streak, current_model_streak)

        if primary_datasets[idx] is not None and primary_datasets[idx] == primary_datasets[idx - 1]:
            current_dataset_streak += 1
        else:
            current_dataset_streak = 1
        max_dataset_streak = max(max_dataset_streak, current_dataset_streak)

    global_run_ids = {value for value in run_ids if value is not None}
    global_datasets: set[str] = set()
    for ds_set in dataset_sets:
        global_datasets.update(ds_set)

    return {
        "window_size": int(real_window),
        "num_samples": len(samples),
        "min_unique_model_run_ids_in_window": int(min_unique_run_ids),
        "min_unique_datasets_in_window": int(min_unique_datasets),
        "global_unique_model_run_ids": len(global_run_ids),
        "global_unique_datasets": len(global_datasets),
        "max_model_run_streak": int(max_model_streak),
        "max_primary_dataset_streak": int(max_dataset_streak),
    }


def _consume_samples(
    collector: CollectorService,
    dataset: SharedModelDataset,
    need: int,
    timeout_seconds: int,
    poll_sleep: float,
    step_start: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    step = step_start
    started = time.time()

    while len(items) < need:
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Timeout while consuming {need} samples, got {len(items)}")

        if collector.is_async_mode:
            process = getattr(collector, "_process", None)
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    "Async collector process exited unexpectedly during smoke run. "
                    "Check collector subprocess traceback in logs."
                )

        if not collector.is_async_mode:
            dataset.maybe_collect(step)

        sample = dataset.try_next_sample()
        if sample is None:
            time.sleep(poll_sleep)
            step += 1
            continue

        items.append(sample)
        step += 1

    return items, step


def _run_smoke(
    cfg: DictConfig,
    target_samples: int,
    timeout_seconds: int,
    poll_sleep: float,
    predownload: bool,
    weight_preview_rows: int,
    weight_preview_cols: int,
    turnover_probe_samples: int,
    turnover_probe_timeout_seconds: int,
    skip_turnover_probe: bool,
    dump_first_weight: str | None,
    diversity_window: int,
) -> dict[str, Any]:
    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    model_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()

    consumed = 0
    first_sample_summary: dict[str, Any] | None = None
    all_consumed_samples: list[Any] = []
    started = time.time()
    step = 0
    turnover_report: dict[str, Any] | None = None

    try:
        if predownload:
            collector.predownload_models()

        collector.start()

        phase_items, step = _consume_samples(
            collector=collector,
            dataset=dataset,
            need=target_samples,
            timeout_seconds=timeout_seconds,
            poll_sleep=poll_sleep,
            step_start=step,
        )

        for sample in phase_items:
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
            all_consumed_samples.append(sample)

        if (
            str(cfg.streaming.mode).lower() != "none"
            and int(turnover_probe_samples) > 0
            and not skip_turnover_probe
        ):
            before = _streaming_chunk_snapshot(cfg) or {"ready_count": collector.cache_size(), "ready_chunk_ids": []}
            probe_items, step = _consume_samples(
                collector=collector,
                dataset=dataset,
                need=int(turnover_probe_samples),
                timeout_seconds=int(turnover_probe_timeout_seconds),
                poll_sleep=poll_sleep,
                step_start=step,
            )
            after = _streaming_chunk_snapshot(cfg) or {"ready_count": collector.cache_size(), "ready_chunk_ids": []}

            before_ids = set(before.get("ready_chunk_ids", []))
            after_ids = set(after.get("ready_chunk_ids", []))
            new_ids = sorted(after_ids - before_ids)
            removed_ids = sorted(before_ids - after_ids)
            observed_run_ids = sorted(
                {
                    int(item.meta.get("model_run_id"))
                    for item in probe_items
                    if item.meta.get("model_run_id") is not None
                }
            )
            all_consumed_samples.extend(probe_items)

            turnover_ok = bool(new_ids or removed_ids or after.get("ready_count") != before.get("ready_count"))
            turnover_report = {
                "executed": True,
                "probe_samples": int(turnover_probe_samples),
                "before_ready_count": int(before.get("ready_count", 0)),
                "after_ready_count": int(after.get("ready_count", 0)),
                "new_ready_chunk_ids_count": len(new_ids),
                "removed_ready_chunk_ids_count": len(removed_ids),
                "new_ready_chunk_ids_head": new_ids[:5],
                "removed_ready_chunk_ids_head": removed_ids[:5],
                "observed_model_run_ids_count": len(observed_run_ids),
                "observed_model_run_ids_head": observed_run_ids[:10],
                "turnover_ok": turnover_ok,
            }

            if not turnover_ok:
                raise RuntimeError(
                    "Streaming turnover probe failed: ready chunk set did not change after additional consumption. "
                    "Try smaller chunk sizes or larger turnover-probe-samples."
                )

    finally:
        dataset.close()
        collector.shutdown()

    if consumed <= 0:
        raise RuntimeError("Smoke run produced zero samples")

    diversity_report = _diversity_report(all_consumed_samples, window_size=int(diversity_window))

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
        "diversity": diversity_report,
        "turnover_probe": turnover_report,
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
            turnover_probe_samples=int(args.turnover_probe_samples),
            turnover_probe_timeout_seconds=int(args.turnover_probe_timeout_seconds),
            skip_turnover_probe=bool(args.skip_turnover_probe),
            dump_first_weight=args.dump_first_weight,
            diversity_window=int(args.diversity_window),
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
