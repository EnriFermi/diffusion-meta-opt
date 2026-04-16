from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf, open_dict


HELDOUT_DATASET_MODELS: dict[str, tuple[str, ...]] = {
    "bigearthnet": ("vit_large_p16_224", "siglip_so400m_p14_384"),
    "chexpert": ("vit_large_p16_224", "clip_vit_l14"),
    "flickr30k": ("clip_vit_l14", "vit_base_p16_224"),
    "food101": ("siglip_so400m_p14_384", "vit_large_p16_224"),
    "openimages_v7": ("detr_resnet50", "clip_vit_l14"),
    "pascal_voc_2012": ("segformer_b5_cityscapes", "detr_resnet50"),
    "rvl_cdip": ("donut_rvlcdip", "trocr_large_printed"),
    "sun397": ("vit_large_p16_224", "clip_vit_l14"),
}


METRIC_NAMES: tuple[str, ...] = (
    "total_loss",
    "behavioral_loss",
    "behavioral_operator",
    "behavioral_dir",
    "behavioral_scale",
    "structural_loss",
    "struct_dir",
    "struct_scale",
    "struct_rec",
    "struct_rel",
    "kl_loss",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def env_path(name: str, default: str | Path) -> Path:
    raw = os.environ.get(name)
    value = str(default) if raw is None or str(raw).strip() == "" else str(raw).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path


def promote_run_profile_to_root(cfg: DictConfig) -> None:
    run_profiles_cfg = cfg.get("run_profiles")
    if not isinstance(run_profiles_cfg, (dict, DictConfig)):
        return

    expected_sections = (
        "data",
        "collector",
        "streaming",
        "train",
        "model",
        "training_artifacts",
        "logging",
        "hf",
        "models",
    )
    with open_dict(cfg):
        for section in expected_sections:
            if section in cfg:
                continue
            if section in run_profiles_cfg:
                cfg[section] = run_profiles_cfg[section]


def sanitize_programmatic_hydra_logging(cfg: DictConfig, *, role: str) -> None:
    """Remove `${hydra:...}` logging interpolations when scripts use compose()."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    with open_dict(cfg):
        if "logging" not in cfg or cfg.logging is None:
            cfg.logging = {}
        project_name = str(cfg.logging.get("project_name", "diffusion_meta_opt"))
        log_dir = str(cfg.logging.get("dir", "logs"))
        file_name = f"{project_name}_{role}_{timestamp}.log"
        cfg.logging.file_name = file_name
        cfg.logging.file_path = str(Path(log_dir) / file_name)


def apply_heldout_log_dir(cfg: DictConfig, *, root_dir: str | Path) -> Path:
    default_log_dir = Path(root_dir).expanduser().parent / "logs"
    log_dir = env_path("HELDOUT_LOG_DIR", default_log_dir)
    with open_dict(cfg):
        if "training_artifacts" not in cfg or cfg.training_artifacts is None:
            cfg.training_artifacts = {}
        if "logging" not in cfg or cfg.logging is None:
            cfg.logging = {}
        cfg.training_artifacts.logs_dir = str(log_dir)
        cfg.logging.dir = str(log_dir)
    return log_dir


def apply_heldout_data_profile(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.data.enabled_datasets = list(HELDOUT_DATASET_MODELS.keys())
        cfg.data.dataset_overrides = {
            dataset_name: {
                "models": list(model_names),
                "sampling_weight": 1.0,
            }
            for dataset_name, model_names in HELDOUT_DATASET_MODELS.items()
        }


def apply_offline_dataset_defaults(cfg: DictConfig, *, root_dir: str | Path) -> None:
    with open_dict(cfg):
        cfg.train.offline_dataset.enabled = True
        cfg.train.offline_dataset.root_dir = str(root_dir)
        cfg.train.offline_dataset.shuffle_chunks = True
        cfg.train.offline_dataset.shuffle_records_within_chunk = True
        cfg.train.offline_dataset.repeat = False
        cfg.train.offline_dataset.weight_cache_size = int(cfg.train.offline_dataset.get("weight_cache_size", 64))
        if "sampling" not in cfg.train.offline_dataset or cfg.train.offline_dataset.sampling is None:
            cfg.train.offline_dataset.sampling = {}
        cfg.train.offline_dataset.sampling.mode = "balanced"
        cfg.train.offline_dataset.sampling.group_keys = ["dataset", "model"]
        cfg.train.offline_dataset.sampling.window_size_records = int(
            cfg.train.offline_dataset.sampling.get("window_size_records", 2048)
        )
        cfg.train.offline_dataset.sampling.max_records_per_chunk_round = int(
            cfg.train.offline_dataset.sampling.get("max_records_per_chunk_round", 8)
        )
        cfg.train.offline_dataset.sampling.x_chunk_cache_size = int(
            cfg.train.offline_dataset.sampling.get("x_chunk_cache_size", 4)
        )


def apply_builder_defaults_from_env(cfg: DictConfig, *, root_dir: str | Path) -> None:
    target_size_bytes = env_int("HELDOUT_TARGET_SIZE_BYTES", 0)
    target_size_gb = env_float("HELDOUT_TARGET_SIZE_GB", 0.0)
    if target_size_bytes <= 0 and target_size_gb <= 0.0:
        target_size_gb = 20.0

    with open_dict(cfg):
        if "builder" not in cfg.train.offline_dataset or cfg.train.offline_dataset.builder is None:
            cfg.train.offline_dataset.builder = {}
        builder = cfg.train.offline_dataset.builder
        cfg.train.offline_dataset.root_dir = str(root_dir)
        builder.target_size_bytes = int(target_size_bytes)
        builder.target_size_gb = float(target_size_gb)
        builder.overwrite_existing = env_bool("HELDOUT_OVERWRITE", False)
        builder.enforce_stage_compatibility = env_bool("HELDOUT_ENFORCE_STAGE_COMPATIBILITY", False)
        builder.x_chunk_size_records = env_int("HELDOUT_X_CHUNK_SIZE_RECORDS", 128)
        builder.max_samples_per_source = env_int("HELDOUT_MAX_SAMPLES_PER_SOURCE", 0)
        builder.max_seen_samples = env_int("HELDOUT_MAX_SEEN_SAMPLES", 0)
        builder.log_every_seen_samples = env_int("HELDOUT_LOG_EVERY_SEEN_SAMPLES", 1000)
        builder.heldout_records_per_pair = env_int("HELDOUT_RECORDS_PER_PAIR", 1024)


def primary_dataset_name(meta: Mapping[str, Any] | None) -> str:
    if not isinstance(meta, Mapping):
        return "<unknown_dataset>"
    image_meta = meta.get("image_meta", [])
    if not isinstance(image_meta, list):
        return "<unknown_dataset>"
    dataset_names = sorted(
        {
            str(item.get("dataset_name", "")).strip()
            for item in image_meta
            if isinstance(item, Mapping) and str(item.get("dataset_name", "")).strip()
        }
    )
    return dataset_names[0] if dataset_names else "<unknown_dataset>"


def sample_pair(sample: Any) -> tuple[str, str]:
    dataset_name = primary_dataset_name(getattr(sample, "meta", {}) or {})
    model_name = str(getattr(sample, "model_name", "") or "").strip() or "<unknown_model>"
    return dataset_name, model_name


def allowed_pairs() -> set[tuple[str, str]]:
    return {
        (dataset_name, model_name)
        for dataset_name, model_names in HELDOUT_DATASET_MODELS.items()
        for model_name in model_names
    }


def expected_datasets() -> set[str]:
    return set(HELDOUT_DATASET_MODELS.keys())


def expected_models() -> set[str]:
    return {model_name for model_names in HELDOUT_DATASET_MODELS.values() for model_name in model_names}


def pair_label(dataset_name: str, model_name: str) -> str:
    return f"{dataset_name}::{model_name}"


def coverage_report(
    *,
    observed_datasets: Iterable[str],
    observed_models: Iterable[str],
    observed_pairs: Iterable[tuple[str, str]],
    skipped: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    expected_dataset_set = expected_datasets()
    expected_model_set = expected_models()
    expected_pair_set = allowed_pairs()

    observed_dataset_set = {str(item) for item in observed_datasets if str(item)}
    observed_model_set = {str(item) for item in observed_models if str(item)}
    observed_pair_set = {(str(dataset), str(model)) for dataset, model in observed_pairs}

    missing_datasets = sorted(expected_dataset_set - observed_dataset_set)
    missing_models = sorted(expected_model_set - observed_model_set)
    missing_pairs = sorted(expected_pair_set - observed_pair_set)
    unexpected_datasets = sorted(observed_dataset_set - expected_dataset_set)
    unexpected_models = sorted(observed_model_set - expected_model_set)
    unexpected_pairs = sorted(observed_pair_set - expected_pair_set)
    skipped_counts = {str(key): int(value) for key, value in dict(skipped or {}).items()}
    skipped_total = int(sum(max(0, value) for value in skipped_counts.values()))

    return {
        "ok": not missing_datasets
        and not missing_models
        and not missing_pairs
        and not unexpected_datasets
        and not unexpected_models
        and not unexpected_pairs
        and skipped_total == 0,
        "expected_dataset_count": int(len(expected_dataset_set)),
        "observed_dataset_count": int(len(observed_dataset_set & expected_dataset_set)),
        "missing_datasets": missing_datasets,
        "unexpected_datasets": unexpected_datasets,
        "expected_model_count": int(len(expected_model_set)),
        "observed_model_count": int(len(observed_model_set & expected_model_set)),
        "missing_models": missing_models,
        "unexpected_models": unexpected_models,
        "expected_pair_count": int(len(expected_pair_set)),
        "observed_pair_count": int(len(observed_pair_set & expected_pair_set)),
        "missing_pairs": [pair_label(dataset, model) for dataset, model in missing_pairs],
        "unexpected_pairs": [pair_label(dataset, model) for dataset, model in unexpected_pairs],
        "skipped": skipped_counts,
        "skipped_total": skipped_total,
    }


def tensor_to_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


@dataclass
class MetricAccumulator:
    weight_sum: float = 0.0
    record_count: int = 0
    slice_count: int = 0
    metric_sums: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in METRIC_NAMES})

    def update(self, metrics: Mapping[str, float], *, weight: int, records: int = 1) -> None:
        safe_weight = max(1, int(weight))
        self.weight_sum += float(safe_weight)
        self.record_count += int(records)
        self.slice_count += safe_weight
        for name in METRIC_NAMES:
            self.metric_sums[name] = self.metric_sums.get(name, 0.0) + float(metrics.get(name, 0.0)) * float(safe_weight)

    def mean_payload(self, **labels: Any) -> dict[str, Any]:
        denom = max(1.0, float(self.weight_sum))
        payload: dict[str, Any] = dict(labels)
        payload["records"] = int(self.record_count)
        payload["slices"] = int(self.slice_count)
        for name in METRIC_NAMES:
            payload[name] = float(self.metric_sums.get(name, 0.0)) / denom
        return payload


class GroupedMetrics:
    def __init__(self) -> None:
        self.global_acc = MetricAccumulator()
        self.by_model: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
        self.by_dataset: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
        self.by_pair: dict[tuple[str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
        self.by_layer: dict[tuple[str, str, str], MetricAccumulator] = defaultdict(MetricAccumulator)

    def update(
        self,
        *,
        dataset_name: str,
        model_name: str,
        layer_name: str,
        metrics: Mapping[str, float],
        weight: int,
    ) -> None:
        self.global_acc.update(metrics, weight=weight)
        self.by_dataset[dataset_name].update(metrics, weight=weight)
        self.by_model[model_name].update(metrics, weight=weight)
        self.by_pair[(dataset_name, model_name)].update(metrics, weight=weight)
        self.by_layer[(dataset_name, model_name, layer_name)].update(metrics, weight=weight)

    @staticmethod
    def _macro(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
        if not rows:
            return {"group": label, "groups": 0, **{name: 0.0 for name in METRIC_NAMES}}
        payload: dict[str, Any] = {"group": label, "groups": int(len(rows))}
        for name in METRIC_NAMES:
            payload[name] = float(sum(float(row.get(name, 0.0)) for row in rows)) / float(len(rows))
        return payload

    def payload(self) -> dict[str, Any]:
        by_model = [acc.mean_payload(model=name) for name, acc in sorted(self.by_model.items())]
        by_dataset = [acc.mean_payload(dataset=name) for name, acc in sorted(self.by_dataset.items())]
        by_pair = [
            acc.mean_payload(dataset=dataset_name, model=model_name)
            for (dataset_name, model_name), acc in sorted(self.by_pair.items())
        ]
        by_layer = [
            acc.mean_payload(dataset=dataset_name, model=model_name, layer=layer_name)
            for (dataset_name, model_name, layer_name), acc in sorted(self.by_layer.items())
        ]
        return {
            "global": self.global_acc.mean_payload(group="micro_global"),
            "macro": {
                "model": self._macro(by_model, "macro_model"),
                "dataset": self._macro(by_dataset, "macro_dataset"),
                "dataset_model_pair": self._macro(by_pair, "macro_dataset_model_pair"),
            },
            "by_model": by_model,
            "by_dataset": by_dataset,
            "by_dataset_model_pair": by_pair,
            "by_layer": by_layer,
        }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def finite_metrics(metrics: Mapping[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def resolved_cfg_snapshot(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    return payload if isinstance(payload, dict) else {}


def counter_to_rows(counter: Counter[tuple[str, str]] | Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = sum(counter.values())
    for key, count in counter.most_common():
        if isinstance(key, tuple):
            row = {f"key_{idx}": value for idx, value in enumerate(key)}
        else:
            row = {"key": key}
        row["count"] = int(count)
        row["share"] = float(count) / max(1.0, float(total))
        rows.append(row)
    return rows
