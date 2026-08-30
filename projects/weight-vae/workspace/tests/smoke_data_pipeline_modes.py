from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.shared.collector_service import CollectorService
from dataset.shared.compatibility_index import CompatibilityIndex, normalize_device, resolve_train_device
from dataset.shared.atomizer import atomize
from dataset.shared.types import MixedImageMeta
from dataset.models.model_pool import ModelPool
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


class PairCoverageError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check the full data pipeline in one of modes: none, local_disk, s3_bridge"
    )
    parser.add_argument("--mode", choices=["none", "local_disk", "s3_bridge"], required=True)
    parser.add_argument(
        "--data-profile",
        default=DEFAULT_FULL_DATASET_PROFILE,
        help=(
            "Hydra data profile name from conf/data/<profile>.yaml "
            "or conf/data_collection_runtime/data_profiles/<profile>.yaml "
            "(default: streaming-safe set without flickr30k)"
        ),
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
        "--pair-audit",
        action="store_true",
        help=(
            "Deterministic one-shot check for every compatible dataset-model pair. "
            "Always writes a report file with exactly one row per pair."
        ),
    )
    parser.add_argument(
        "--pair-audit-max-seconds",
        type=int,
        default=600,
        help="Hard wall-clock budget for deterministic pair-audit mode.",
    )
    parser.add_argument(
        "--pair-audit-sample-timeout-seconds",
        type=float,
        default=10.0,
        help="Per-dataset sample fetch timeout used by pair-audit worker overrides.",
    )
    parser.add_argument(
        "--pair-audit-report-path",
        default="",
        help="Output path for pair-audit JSON report. If empty, path is auto-generated under ./data/reports/.",
    )
    parser.add_argument(
        "--pair-coverage-timeout-seconds",
        type=int,
        default=300,
        help="Additional timeout budget for ensuring every compatible dataset-model pair is observed at least once.",
    )
    parser.add_argument(
        "--require-all-dataset-model-pairs",
        dest="require_all_dataset_model_pairs",
        action="store_true",
        help="Require seeing every compatible dataset-model pair at least once.",
    )
    parser.add_argument(
        "--allow-missing-dataset-model-pairs",
        dest="require_all_dataset_model_pairs",
        action="store_false",
        help="Do not fail if some compatible dataset-model pairs were not observed.",
    )
    parser.set_defaults(require_all_dataset_model_pairs=True)
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
        cfg = compose(config_name="big_vae/train/default", overrides=overrides)

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


def _normalize_profile_name(profile_name: str) -> str:
    name = str(profile_name).strip()
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    return name


def _resolve_data_profile(path_or_name: str) -> tuple[Path, str]:
    profile_name = _normalize_profile_name(path_or_name)
    if not profile_name:
        raise ValueError("data profile name is empty")

    root_conf = _project_root() / "conf"
    conf_data_path = root_conf / "data" / f"{profile_name}.yaml"
    if conf_data_path.exists():
        # `data` is not in root defaults anymore, so append the group.
        return conf_data_path, f"+data={profile_name}"

    runtime_data_path = root_conf / "data_collection_runtime" / "data_profiles" / f"{profile_name}.yaml"
    if runtime_data_path.exists():
        return runtime_data_path, f"data_collection_runtime/data_profiles@data={profile_name}"

    raise FileNotFoundError(
        f"Unknown data profile '{path_or_name}'. "
        f"Tried {conf_data_path} and {runtime_data_path}"
    )


def _build_overrides(args: argparse.Namespace) -> list[str]:
    profile_path, profile_override = _resolve_data_profile(str(args.data_profile))
    run_tag = str(args.run_tag).strip() if args.run_tag else f"run_{int(time.time())}"
    cli_datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    if cli_datasets:
        datasets = cli_datasets
    else:
        datasets = _resolve_profile_enabled_datasets(profile_path)
        if not datasets:
            datasets = list(DEFAULT_FULL_DATASET_LIST)

    overrides: list[str] = [
        profile_override,
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
        if bool(args.pair_audit):
            sample_timeout = max(1.0, float(args.pair_audit_sample_timeout_seconds))
            overrides.append(f"+data.dataset_overrides.{dataset_name}.worker.get_timeout_s={sample_timeout}")
            overrides.append(f"+data.dataset_overrides.{dataset_name}.worker.startup_get_timeout_s={sample_timeout}")
            overrides.append(f"+data.dataset_overrides.{dataset_name}.worker.request_timeout_s={max(1, int(sample_timeout))}")
            overrides.append(f"+data.dataset_overrides.{dataset_name}.worker.max_worker_restarts=1")

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


def _resolve_profile_enabled_datasets(profile_name: Path | str) -> list[str]:
    profile_path = Path(profile_name)
    try:
        profile_cfg = OmegaConf.load(profile_path)
    except Exception:
        return []

    enabled = profile_cfg.get("enabled_datasets")
    if not isinstance(enabled, (list, tuple, ListConfig)):
        return []

    datasets: list[str] = []
    for value in enabled:
        name = str(value).strip()
        if not name:
            continue
        datasets.append(name)

    return datasets


def _resolve_pair_audit_report_path(args: argparse.Namespace) -> Path:
    raw = str(args.pair_audit_report_path).strip()
    if raw:
        out_path = Path(raw)
    else:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        out_path = Path(args.data_root) / "reports" / f"pair_audit_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _write_report_file(path: Path, payload: dict[str, Any]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:
        fallback_dir = Path("./data/reports")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        fallback_path = fallback_dir / f"pair_audit_fallback_{ts}.json"
        payload = dict(payload)
        payload["report_write_error"] = traceback.format_exc()
        fallback_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return fallback_path


def _extract_dataset_runtime_error(raw_pool: Any, dataset_name: str) -> str:
    dataset_obj = getattr(raw_pool, "datasets", {}).get(dataset_name)
    if dataset_obj is None:
        return "dataset object is missing in RawDatasetPool"

    last_worker_error = getattr(dataset_obj, "_last_worker_error", None)
    if isinstance(last_worker_error, str) and last_worker_error.strip():
        return last_worker_error

    try:
        stats = dataset_obj.stats()
    except Exception:
        stats = {}
    return f"no worker error details; stats={stats}"


def _run_pair_audit(cfg: DictConfig, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    max_seconds = max(1.0, float(args.pair_audit_max_seconds))
    report_path = _resolve_pair_audit_report_path(args)
    sample_timeout_s = max(1.0, float(args.pair_audit_sample_timeout_seconds))

    compat = CompatibilityIndex(cfg)
    expected_pairs = sorted(
        {
        (str(dataset_name), str(model_name))
        for model_name in compat.get_models()
        for dataset_name in compat.get_datasets_for_model(model_name)
        }
    )

    pair_rows: dict[tuple[str, str], dict[str, Any]] = {
        pair: {
            "dataset_name": pair[0],
            "model_name": pair[1],
            "status": "timeout",
            "attempted": False,
            "elapsed_s": None,
            "sample_timeout_s": float(sample_timeout_s),
            "error": "Not attempted",
            "traceback": None,
            "num_layer_records": 0,
            "num_shared_samples": 0,
        }
        for pair in expected_pairs
    }

    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_plain, dict):
        cfg_plain = {}
    collector_cfg_plain = cfg_plain.get("collector", {})
    if not isinstance(collector_cfg_plain, dict):
        collector_cfg_plain = {}

    runtime_device = normalize_device(collector_cfg_plain.get("device"))
    if runtime_device is None:
        runtime_device = resolve_train_device(cfg_plain)

    atom_cfg = {}
    if isinstance(collector_cfg_plain.get("layer_output_splitting"), dict):
        atom_cfg = dict(collector_cfg_plain.get("layer_output_splitting") or {})
    if "xy_samples_random_slice" not in atom_cfg:
        atom_cfg["xy_samples_random_slice"] = 64

    model_pool: ModelPool | None = None
    raw_pool: Any | None = None
    deadline_hit = False
    global_error_traceback: str | None = None
    global_error_message: str | None = None

    try:
        model_pool = ModelPool(
            global_cfg=cfg,
            model_cfgs=compat.get_model_cfgs(),
            device_override=runtime_device,
            max_loaded_models=max(1, int(collector_cfg_plain.get("max_loaded_models", 1))),
            runtime_local_only=bool(collector_cfg_plain.get("runtime_local_only", False)),
            release_device_on_unload=bool(collector_cfg_plain.get("release_device_on_unload", True)),
            empty_cuda_cache_on_unload=bool(collector_cfg_plain.get("empty_cuda_cache_on_unload", True)),
        )

        from dataset.shared.raw_dataset_pool import RawDatasetPool

        raw_pool = RawDatasetPool(cfg=cfg, index=compat)
        raw_pool.start()

        if bool(args.predownload):
            model_pool.predownload_models(sorted(compat.get_models()))

        ordered_pairs = sorted(expected_pairs, key=lambda item: (item[1], item[0]))
        for dataset_name, model_name in ordered_pairs:
            if (time.time() - started) >= max_seconds:
                deadline_hit = True
                break

            row = pair_rows[(dataset_name, model_name)]
            row["attempted"] = True
            pair_started = time.time()
            try:
                pil_batch, source_ids, _ = raw_pool.get_pil_batch(dataset_name=dataset_name, n=1)
                if not pil_batch:
                    runtime_error = _extract_dataset_runtime_error(raw_pool=raw_pool, dataset_name=dataset_name)
                    raise RuntimeError(
                        f"Dataset returned zero images for dataset={dataset_name}; "
                        f"sample_timeout_s={sample_timeout_s}. last_error={runtime_error}"
                    )

                layer_records = model_pool.run(model_name=model_name, pil_batch=pil_batch[:1])
                if not layer_records:
                    raise RuntimeError(f"Model run produced zero layer records for model={model_name}")

                source_id = source_ids[0] if source_ids else "unknown"
                image_meta_list = [MixedImageMeta(dataset_name=dataset_name, source_id=source_id)]
                shared_samples_count = 0
                for layer_record in layer_records:
                    shared_samples_count += sum(
                        1
                        for _ in atomize(
                            layer_record=layer_record,
                            atom_cfg=atom_cfg,
                            image_meta_list=image_meta_list,
                            model_run_id=1,
                        )
                    )
                    if shared_samples_count > 0:
                        break

                if shared_samples_count <= 0:
                    raise RuntimeError(
                        f"Atomizer produced zero SharedSample for pair dataset={dataset_name} model={model_name}"
                    )

                row["status"] = "success"
                row["error"] = None
                row["traceback"] = None
                row["num_layer_records"] = int(len(layer_records))
                row["num_shared_samples"] = int(shared_samples_count)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error"] = str(exc)
                row["traceback"] = traceback.format_exc()
                row["num_layer_records"] = int(row.get("num_layer_records", 0) or 0)
                row["num_shared_samples"] = int(row.get("num_shared_samples", 0) or 0)
            finally:
                row["elapsed_s"] = round(max(0.0, float(time.time() - pair_started)), 3)
    except Exception as exc:  # noqa: BLE001
        global_error_message = str(exc)
        global_error_traceback = traceback.format_exc()
    finally:
        if raw_pool is not None:
            try:
                raw_pool.shutdown()
            except Exception:
                pass
        if model_pool is not None:
            try:
                model_pool.unload_all()
            except Exception:
                pass

    elapsed_total = max(0.0, float(time.time() - started))

    for pair in expected_pairs:
        row = pair_rows[pair]
        if row["status"] != "timeout":
            continue
        if global_error_message is not None:
            row["status"] = "failed"
            row["error"] = f"Pair audit aborted by global error: {global_error_message}"
            row["traceback"] = global_error_traceback
        elif deadline_hit:
            row["error"] = "Global pair-audit deadline exceeded before attempt"
            row["traceback"] = "Traceback (most recent call last):\nTimeoutError: global pair-audit deadline exceeded"
        else:
            row["error"] = "Pair was not attempted"
            row["traceback"] = "Traceback (most recent call last):\nRuntimeError: pair was not attempted"

    rows_ordered = [pair_rows[pair] for pair in sorted(pair_rows.keys())]
    success_count = sum(1 for row in rows_ordered if row["status"] == "success")
    failed_count = sum(1 for row in rows_ordered if row["status"] == "failed")
    timeout_count = sum(1 for row in rows_ordered if row["status"] == "timeout")

    payload = {
        "ok": failed_count == 0 and timeout_count == 0 and success_count == len(rows_ordered),
        "mode": "pair_audit",
        "max_seconds": float(max_seconds),
        "pair_sample_timeout_seconds": float(sample_timeout_s),
        "elapsed_seconds": round(elapsed_total, 3),
        "deadline_hit": bool(deadline_hit),
        "global_error": global_error_message,
        "total_pairs": int(len(rows_ordered)),
        "success_count": int(success_count),
        "failed_count": int(failed_count),
        "timeout_count": int(timeout_count),
        "report_path": str(report_path.resolve()),
        "pairs": rows_ordered,
    }

    written_path = _write_report_file(report_path, payload)
    payload["report_path"] = str(written_path.resolve())
    return payload


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
    timeout_seconds: float,
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
    pair_coverage_timeout_seconds: int,
    require_all_dataset_model_pairs: bool,
) -> dict[str, Any]:
    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)

    model_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_first_seen_s: dict[tuple[str, str], float] = {}
    expected_pairs: set[tuple[str, str]] = {
        (str(dataset_name), str(model_name))
        for model_name in collector.compat_index.get_models()
        for dataset_name in collector.compat_index.get_datasets_for_model(model_name)
    }

    consumed = 0
    first_sample_summary: dict[str, Any] | None = None
    all_consumed_samples: list[Any] = []
    started = time.time()
    step = 0
    turnover_report: dict[str, Any] | None = None
    pair_coverage_elapsed_s = 0.0

    def _process_sample(sample: Any) -> None:
        nonlocal consumed, first_sample_summary
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

        model_name = str(sample.model_name)
        model_counts[model_name] += 1
        layer_counts[str(sample.layer_name)] += 1

        sample_dataset_names: set[str] = set()
        for item in sample.meta.get("image_meta", []):
            ds_name = item.get("dataset_name")
            if ds_name:
                ds_name_text = str(ds_name)
                dataset_counts[ds_name_text] += 1
                sample_dataset_names.add(ds_name_text)

        for ds_name_text in sample_dataset_names:
            pair = (ds_name_text, model_name)
            if pair_counts[pair] == 0:
                pair_first_seen_s[pair] = max(0.0, float(time.time() - started))
            pair_counts[pair] += 1

        consumed += 1
        all_consumed_samples.append(sample)

    def _missing_pairs() -> list[tuple[str, str]]:
        observed = set(pair_counts.keys())
        return sorted(expected_pairs - observed)

    def _pair_timing_payload(missing_pairs: list[tuple[str, str]], timed_out: bool) -> list[dict[str, Any]]:
        missing_set = set(missing_pairs)
        payload: list[dict[str, Any]] = []
        for dataset_name, model_name in sorted(expected_pairs):
            key = (dataset_name, model_name)
            first_seen = pair_first_seen_s.get(key)
            payload.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "seen": first_seen is not None,
                    "first_seen_s": round(float(first_seen), 3) if first_seen is not None else None,
                    "timed_out": bool(timed_out and key in missing_set),
                }
            )
        return payload

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
            _process_sample(sample)

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
            for sample in probe_items:
                _process_sample(sample)

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

        if require_all_dataset_model_pairs:
            coverage_started = time.time()
            coverage_timeout_hit = False
            missing_pairs = _missing_pairs()
            while missing_pairs:
                elapsed = time.time() - coverage_started
                remaining = float(pair_coverage_timeout_seconds) - elapsed
                if remaining <= 0:
                    coverage_timeout_hit = True
                    break

                extra_need = max(1, min(16, len(missing_pairs)))
                extra_items, step = _consume_samples(
                    collector=collector,
                    dataset=dataset,
                    need=extra_need,
                    timeout_seconds=remaining,
                    poll_sleep=poll_sleep,
                    step_start=step,
                )
                for sample in extra_items:
                    _process_sample(sample)
                missing_pairs = _missing_pairs()

            pair_coverage_elapsed_s = max(0.0, float(time.time() - coverage_started))
            if missing_pairs:
                diagnostics = {
                    "pair_coverage_timeout_seconds": float(pair_coverage_timeout_seconds),
                    "pair_coverage_elapsed_seconds": round(pair_coverage_elapsed_s, 3),
                    "dataset_model_pairs_expected_count": int(len(expected_pairs)),
                    "dataset_model_pairs_observed_count": int(len(pair_counts)),
                    "dataset_model_pairs_missing_count": int(len(missing_pairs)),
                    "dataset_model_pairs_missing": [
                        {"dataset_name": dataset_name, "model_name": model_name}
                        for dataset_name, model_name in missing_pairs
                    ],
                    "dataset_model_pair_counts": [
                        {"dataset_name": dataset_name, "model_name": model_name, "count": int(count)}
                        for (dataset_name, model_name), count in sorted(pair_counts.items())
                    ],
                    "dataset_model_pair_timings": _pair_timing_payload(
                        missing_pairs=missing_pairs,
                        timed_out=coverage_timeout_hit,
                    ),
                }
                raise PairCoverageError(
                    "Dataset-model coverage check failed: "
                    f"observed_pairs={len(pair_counts)}/{len(expected_pairs)}, "
                    f"missing_pairs={len(missing_pairs)}. "
                    "Increase --target-samples and/or --pair-coverage-timeout-seconds "
                    "or disable strict check with --allow-missing-dataset-model-pairs.",
                    diagnostics=diagnostics,
                )

    finally:
        dataset.close()
        collector.shutdown()

    if consumed <= 0:
        raise RuntimeError("Smoke run produced zero samples")

    diversity_report = _diversity_report(all_consumed_samples, window_size=int(diversity_window))
    missing_pairs_final = _missing_pairs()
    pair_counts_serialized = [
        {"dataset_name": dataset_name, "model_name": model_name, "count": int(count)}
        for (dataset_name, model_name), count in sorted(pair_counts.items())
    ]

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
        "dataset_model_pair_coverage_required": bool(require_all_dataset_model_pairs),
        "pair_coverage_timeout_seconds": float(pair_coverage_timeout_seconds),
        "pair_coverage_elapsed_seconds": round(float(pair_coverage_elapsed_s), 3),
        "dataset_model_pairs_expected_count": int(len(expected_pairs)),
        "dataset_model_pairs_observed_count": int(len(pair_counts)),
        "dataset_model_pairs_missing_count": int(len(missing_pairs_final)),
        "dataset_model_pairs_missing": [
            {"dataset_name": dataset_name, "model_name": model_name}
            for dataset_name, model_name in missing_pairs_final
        ],
        "dataset_model_pair_counts": pair_counts_serialized,
        "dataset_model_pair_timings": _pair_timing_payload(
            missing_pairs=missing_pairs_final,
            timed_out=False,
        ),
        "collector_stats": collector.stats(),
    }


def main() -> int:
    args = _parse_args()
    try:
        overrides = _build_overrides(args)
        cfg = _load_cfg(overrides)
    except Exception as exc:
        if bool(args.pair_audit):
            report_path = _resolve_pair_audit_report_path(args)
            payload = {
                "ok": False,
                "mode": "pair_audit",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "total_pairs": 0,
                "success_count": 0,
                "failed_count": 0,
                "timeout_count": 0,
                "pairs": [],
                "report_path": str(report_path.resolve()),
            }
            written_path = _write_report_file(report_path, payload)
            payload["report_path"] = str(written_path.resolve())
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        payload = {"ok": False, "error": str(exc)}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg, resolve=True))

    if bool(args.pair_audit):
        result = _run_pair_audit(cfg=cfg, args=args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # Pair-audit always returns structured report instead of throwing.
        return 0

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
            pair_coverage_timeout_seconds=int(args.pair_coverage_timeout_seconds),
            require_all_dataset_model_pairs=bool(args.require_all_dataset_model_pairs),
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
        if isinstance(exc, PairCoverageError):
            payload.update(exc.diagnostics)
        if hint:
            payload["hint"] = hint
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
