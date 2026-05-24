from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset.logging_utils import configure_process_logging
from training.runtime import patch_argparse_lazy_help_for_hydra_py314


patch_argparse_lazy_help_for_hydra_py314()


LOGGER = logging.getLogger("analyze_big_vae_offline_dataset")

_DEPTH_PATTERNS = (
    re.compile(r"(?:^|\.)(?:layers|layer|blocks|block|h|resblocks|encoder_layers|decoder_layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:encoder|decoder)\.(?:layers|layer|blocks|block)\.(\d+)(?:\.|$)"),
)


def _promote_run_profile_to_root(cfg: DictConfig) -> None:
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required offline dataset file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def infer_layer_type(layer_name: str) -> str:
    name = str(layer_name).lower()

    if any(token in name for token in ("query", "q_proj", ".q.", "self.q", "attn.q", ".qkv")):
        return "attn_query"
    if any(token in name for token in ("key", "k_proj", ".k.", "self.k", "attn.k")):
        return "attn_key"
    if any(token in name for token in ("value", "v_proj", ".v.", "self.v", "attn.v")):
        return "attn_value"
    if any(token in name for token in ("out_proj", "output.dense", "attention.output", "attn.proj")):
        return "attn_output"
    if "attn" in name or "attention" in name:
        return "attn_other"

    if any(token in name for token in ("intermediate", "fc1", "mlp.fc1", "gate_proj", "up_proj")):
        return "ffn_up"
    if any(token in name for token in ("output.dense", "fc2", "mlp.fc2", "down_proj")):
        return "ffn_down"

    if "pooler" in name:
        return "pooler"
    if any(token in name for token in ("embed", "embedding")):
        return "embedding"
    if "conv" in name:
        return "conv"
    if any(token in name for token in ("lm_head", "classifier", "score", "head")):
        return "head"
    return "other_linear"


def infer_layer_depth(layer_name: str) -> int | None:
    name = str(layer_name).strip()
    if not name:
        return None
    for pattern in _DEPTH_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            return int(match.group(1))
    return None


def _shape_key(d_in: int, d_out: int) -> str:
    return f"{int(d_in)}x{int(d_out)}"


def _pow2_bucket(value: int) -> str:
    value = int(value)
    if value <= 0:
        return "0"
    lower = 1 << int(math.floor(math.log2(value)))
    upper = max(lower, (lower << 1) - 1)
    return f"{lower}-{upper}"


def _counter_gini(counter: Counter[str]) -> float:
    counts = sorted(int(value) for value in counter.values() if int(value) > 0)
    if not counts:
        return 0.0
    n = len(counts)
    total = float(sum(counts))
    weighted = sum((idx + 1) * value for idx, value in enumerate(counts))
    return float((2.0 * weighted) / (n * total) - (n + 1) / n)


def _counter_report(counter: Counter[str], *, topk: int, item_key: str = "name") -> dict[str, Any]:
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


def _top_items(counter: Counter[str], *, topk: int, item_key: str = "name") -> list[dict[str, Any]]:
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


def _skew_hints(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    model_top1 = float(report["models"]["records"].get("top1_share", 0.0))
    shape_top1 = float(report["shapes"]["records"].get("top1_share", 0.0))
    type_top1 = float(report["layer_types"]["records"].get("top1_share", 0.0))
    unknown_depth = 0.0
    depth_items = report["depths"]["records"].get("items", [])
    for item in depth_items:
        if str(item.get("depth")) == "unknown":
            unknown_depth = float(item.get("share", 0.0))
            break

    if model_top1 >= 0.5:
        hints.append(f"record distribution is model-heavy: top model share={model_top1:.1%}")
    if shape_top1 >= 0.35:
        hints.append(f"record distribution is shape-heavy: top exact shape share={shape_top1:.1%}")
    if type_top1 >= 0.6:
        hints.append(f"record distribution is layer-type-heavy: top type share={type_top1:.1%}")
    if unknown_depth >= 0.5:
        hints.append(f"depth inference is weak for this corpus: unknown depth share={unknown_depth:.1%}")
    return hints


def analyze_big_vae_offline_dataset_root(
    root_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    topk: int = 20,
    per_model_topk: int = 10,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    root = Path(str(root_dir)).expanduser().resolve()
    logger_local = logger or LOGGER
    manifest = _load_json(root / "manifest.json")
    sources_payload = _load_json(root / "sources.json")
    sources_raw = sources_payload.get("sources", [])
    if not isinstance(sources_raw, list):
        raise TypeError(f"Expected 'sources' list in {root / 'sources.json'}")

    sources: list[dict[str, Any]] = [dict(item) for item in sources_raw if isinstance(item, dict)]

    model_sources: Counter[str] = Counter()
    model_records: Counter[str] = Counter()
    shape_sources: Counter[str] = Counter()
    shape_records: Counter[str] = Counter()
    layer_type_sources: Counter[str] = Counter()
    layer_type_records: Counter[str] = Counter()
    depth_sources: Counter[str] = Counter()
    depth_records: Counter[str] = Counter()
    d_in_sources: Counter[str] = Counter()
    d_in_records: Counter[str] = Counter()
    d_out_sources: Counter[str] = Counter()
    d_out_records: Counter[str] = Counter()
    params_sources: Counter[str] = Counter()
    params_records: Counter[str] = Counter()

    model_to_types: dict[str, Counter[str]] = defaultdict(Counter)
    model_to_depths: dict[str, Counter[str]] = defaultdict(Counter)
    model_to_shapes: dict[str, Counter[str]] = defaultdict(Counter)
    model_to_layers: dict[str, Counter[str]] = defaultdict(Counter)

    total_records = 0
    total_sources = 0

    for source in sources:
        model_name = str(source.get("model_name", "") or "").strip() or "<unknown_model>"
        layer_name = str(source.get("layer_name", "") or "").strip() or "<unknown_layer>"
        raw_shape = source.get("weight_shape", [])
        if isinstance(raw_shape, (list, tuple)) and len(raw_shape) >= 2:
            d_in = int(raw_shape[0])
            d_out = int(raw_shape[1])
        else:
            d_in = 0
            d_out = 0
        num_records = max(0, int(source.get("num_records", 0)))
        num_records = max(1, num_records)
        total_records += num_records
        total_sources += 1

        layer_type = infer_layer_type(layer_name)
        depth_value = infer_layer_depth(layer_name)
        depth_label = str(depth_value) if depth_value is not None else "unknown"
        shape_label = _shape_key(d_in=d_in, d_out=d_out)
        d_in_bucket = _pow2_bucket(d_in)
        d_out_bucket = _pow2_bucket(d_out)
        params_bucket = _pow2_bucket(d_in * d_out)

        model_sources[model_name] += 1
        model_records[model_name] += num_records
        shape_sources[shape_label] += 1
        shape_records[shape_label] += num_records
        layer_type_sources[layer_type] += 1
        layer_type_records[layer_type] += num_records
        depth_sources[depth_label] += 1
        depth_records[depth_label] += num_records
        d_in_sources[d_in_bucket] += 1
        d_in_records[d_in_bucket] += num_records
        d_out_sources[d_out_bucket] += 1
        d_out_records[d_out_bucket] += num_records
        params_sources[params_bucket] += 1
        params_records[params_bucket] += num_records

        model_to_types[model_name][layer_type] += num_records
        model_to_depths[model_name][depth_label] += num_records
        model_to_shapes[model_name][shape_label] += num_records
        model_to_layers[model_name][layer_name] += num_records

    report: dict[str, Any] = {
        "root_dir": str(root),
        "manifest_path": str(root / "manifest.json"),
        "sources_path": str(root / "sources.json"),
        "actual_size_bytes": int(manifest.get("actual_size_bytes", 0)),
        "actual_size_gb": float(manifest.get("actual_size_gb", 0.0)),
        "accepted_records": int(manifest.get("accepted_records", total_records)),
        "unique_sources": int(manifest.get("unique_sources", total_sources)),
        "unique_models": int(manifest.get("unique_models", len(model_sources))),
        "unique_layers": int(manifest.get("unique_layers", sum(len(counter) for counter in model_to_layers.values()))),
        "models": {
            "sources": _counter_report(model_sources, topk=topk, item_key="model_name"),
            "records": _counter_report(model_records, topk=topk, item_key="model_name"),
        },
        "layer_types": {
            "sources": _counter_report(layer_type_sources, topk=topk, item_key="layer_type"),
            "records": _counter_report(layer_type_records, topk=topk, item_key="layer_type"),
        },
        "depths": {
            "sources": _counter_report(depth_sources, topk=topk, item_key="depth"),
            "records": _counter_report(depth_records, topk=topk, item_key="depth"),
        },
        "shapes": {
            "sources": _counter_report(shape_sources, topk=topk, item_key="shape"),
            "records": _counter_report(shape_records, topk=topk, item_key="shape"),
            "d_in_buckets_sources": _counter_report(d_in_sources, topk=topk, item_key="d_in_bucket"),
            "d_in_buckets_records": _counter_report(d_in_records, topk=topk, item_key="d_in_bucket"),
            "d_out_buckets_sources": _counter_report(d_out_sources, topk=topk, item_key="d_out_bucket"),
            "d_out_buckets_records": _counter_report(d_out_records, topk=topk, item_key="d_out_bucket"),
            "num_params_buckets_sources": _counter_report(params_sources, topk=topk, item_key="num_params_bucket"),
            "num_params_buckets_records": _counter_report(params_records, topk=topk, item_key="num_params_bucket"),
        },
        "per_model": [],
        "stage_targets": dict(manifest.get("stage_targets", {}) or {}),
    }

    total_record_float = float(total_records) if total_records > 0 else 1.0
    total_source_float = float(total_sources) if total_sources > 0 else 1.0
    for model_name, record_count in model_records.most_common(max(1, int(per_model_topk))):
        source_count = int(model_sources.get(model_name, 0))
        report["per_model"].append(
            {
                "model_name": model_name,
                "num_records": int(record_count),
                "share_records": float(record_count) / total_record_float,
                "num_sources": source_count,
                "share_sources": float(source_count) / total_source_float,
                "unique_layer_types": int(len(model_to_types[model_name])),
                "unique_depths": int(len(model_to_depths[model_name])),
                "unique_shapes": int(len(model_to_shapes[model_name])),
                "top_layer_types": _top_items(model_to_types[model_name], topk=5, item_key="layer_type"),
                "top_depths": _top_items(model_to_depths[model_name], topk=5, item_key="depth"),
                "top_shapes": _top_items(model_to_shapes[model_name], topk=5, item_key="shape"),
                "top_layers": _top_items(model_to_layers[model_name], topk=5, item_key="layer_name"),
            }
        )

    report["skew_hints"] = _skew_hints(report)

    output_root = Path(output_dir).expanduser().resolve() if output_dir is not None else (root / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "diversity_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    logger_local.info("Offline BigVAE diversity report written: %s", report_path)
    return report


def _summary_lines(report: dict[str, Any]) -> list[str]:
    model_items = report["models"]["records"].get("items", [])
    type_items = report["layer_types"]["records"].get("items", [])
    depth_items = report["depths"]["records"].get("items", [])
    shape_items = report["shapes"]["records"].get("items", [])

    lines = [
        (
            "Offline dataset summary: "
            f"size_gb={float(report.get('actual_size_gb', 0.0)):.2f} "
            f"records={int(report.get('accepted_records', 0))} "
            f"sources={int(report.get('unique_sources', 0))} "
            f"models={int(report.get('unique_models', 0))} "
            f"layers={int(report.get('unique_layers', 0))}"
        ),
        (
            "Model diversity: "
            f"top1_share={float(report['models']['records'].get('top1_share', 0.0)):.1%} "
            f"normalized_entropy={float(report['models']['records'].get('normalized_entropy', 1.0)):.3f} "
            f"top_models={[item.get('model_name') for item in model_items[:5]]}"
        ),
        (
            "Layer-type diversity: "
            f"top1_share={float(report['layer_types']['records'].get('top1_share', 0.0)):.1%} "
            f"normalized_entropy={float(report['layer_types']['records'].get('normalized_entropy', 1.0)):.3f} "
            f"top_types={[item.get('layer_type') for item in type_items[:5]]}"
        ),
        (
            "Depth diversity: "
            f"top1_share={float(report['depths']['records'].get('top1_share', 0.0)):.1%} "
            f"normalized_entropy={float(report['depths']['records'].get('normalized_entropy', 1.0)):.3f} "
            f"top_depths={[item.get('depth') for item in depth_items[:5]]}"
        ),
        (
            "Shape diversity: "
            f"top1_share={float(report['shapes']['records'].get('top1_share', 0.0)):.1%} "
            f"normalized_entropy={float(report['shapes']['records'].get('normalized_entropy', 1.0)):.3f} "
            f"top_shapes={[item.get('shape') for item in shape_items[:5]]}"
        ),
    ]
    for hint in report.get("skew_hints", []):
        lines.append(f"Skew hint: {hint}")
    lines.append(f"Report path: {report.get('report_path', '')}")
    return lines


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _promote_run_profile_to_root(cfg)
    log_path = configure_process_logging(cfg=cfg, role="analyze_big_vae_offline_dataset", rank=0, force=True)
    logger = logging.getLogger("analyze_big_vae_offline_dataset")

    offline_cfg = cfg.train.get("offline_dataset", {})
    if not isinstance(offline_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset must be a mapping")
    analysis_cfg = offline_cfg.get("analysis", {})
    if analysis_cfg is None:
        analysis_cfg = {}
    if not isinstance(analysis_cfg, (dict, DictConfig)):
        raise TypeError("train.offline_dataset.analysis must be a mapping")

    root_dir = str(offline_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("train.offline_dataset.root_dir must be set for offline dataset analysis")

    logger.info("Starting offline BigVAE dataset analysis")
    logger.info("Run log file: %s", log_path)
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    report = analyze_big_vae_offline_dataset_root(
        root_dir=root_dir,
        output_dir=str(analysis_cfg.get("output_dir", "")) or None,
        topk=int(analysis_cfg.get("topk", 20)),
        per_model_topk=int(analysis_cfg.get("per_model_topk", 10)),
        logger=logger,
    )
    for line in _summary_lines(report):
        logger.info(line)


if __name__ == "__main__":
    main()
