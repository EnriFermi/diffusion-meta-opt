from __future__ import annotations

"""Source-only conditioning-density factorial for the canonical Weight-AE.

This evaluator deliberately has no target-domain argument or import.  It reuses
the locked source split, source gain panel, held-out ViT-B A/B panel, and two
tilings from ``source_confirmatory_g01_gate.py``.  All target domains remain
sealed regardless of the result.
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import source_confirmatory_g01_gate as base


VARIANT_SPECS: tuple[tuple[str, int | None, int], ...] = (
    ("d2_k1", 2, 1),
    ("d2_k4", 2, 4),
    ("d2_k8", 2, 8),
    ("all_k1", None, 1),
    ("all_k4", None, 4),
    ("all_k8", None, 8),
)
VARIANTS = tuple(value[0] for value in VARIANT_SPECS)
CONTROLS = ("ae_native_c", "ae_zero_c", "identity", "zero")
METHODS = VARIANTS + CONTROLS
DEFAULT_OUTPUT = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_condition_density_gate_20260816"
)
DEFAULT_SPARSE_REFERENCE = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_confirmatory_gate_20260816_clean2"
)


def variant_spec(name: str) -> tuple[int | None, int]:
    for candidate, dataset_cap, record_cap in VARIANT_SPECS:
        if candidate == name:
            return dataset_cap, record_cap
    raise KeyError(name)


def dataset_rank_key(seed: int, source_key: str, dataset: str) -> str:
    # Exact old sparse-template contract.
    return base.stable_hex(seed, "c0_dataset", source_key, dataset)


def record_rank_key(seed: int, ref: base.RecordRef) -> str:
    # Exact old sparse-template contract.
    return base.ref_hash(ref, seed, "c0_record")


def record_identity(ref: base.RecordRef) -> str:
    return f"{ref.source_key}:{ref.chunk_idx}:{ref.record_idx}"


def canonical_json_hash(value: Any) -> str:
    return base.stable_hex(json.dumps(value, sort_keys=True, separators=(",", ":")))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    base.write_json(temporary, value)
    temporary.replace(path)


def select_cache_records(
    refs_by_source: dict[str, list[base.RecordRef]],
    source_best: dict[str, base.RecordRef],
    *,
    structure: dict[str, tuple[int, int]],
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source_key, identity in sorted(source_best.items()):
        ordinal, layer_count = structure[source_key]
        by_dataset: dict[str, list[base.RecordRef]] = defaultdict(list)
        for ref in refs_by_source[source_key]:
            by_dataset[ref.primary_dataset].append(ref)
        datasets = sorted(
            by_dataset,
            key=lambda dataset: dataset_rank_key(seed, source_key, dataset),
        )
        if not datasets:
            raise RuntimeError(f"source has no activation dataset: {source_key}")
        for dataset_rank, dataset in enumerate(datasets):
            refs = sorted(by_dataset[dataset], key=lambda ref: record_rank_key(seed, ref))
            for record_rank, ref in enumerate(refs[:8]):
                selected.append(
                    {
                        "source_key": source_key,
                        "model_name": identity.model_name,
                        "role": identity.role,
                        "native_ordinal": ordinal,
                        "native_layers": layer_count,
                        "native_u": ordinal / (layer_count - 1),
                        "dataset": dataset,
                        "dataset_rank": dataset_rank,
                        "available_distinct_datasets": len(datasets),
                        "record_rank": record_rank,
                        "available_records_for_pair": len(refs),
                        "ref": ref,
                    }
                )
    # Chunk-local I/O order is locked before extraction.  Template hierarchy is
    # reconstructed by explicit ranks, so I/O ordering cannot change weighting.
    selected.sort(key=lambda item: (item["ref"].chunk_idx, item["ref"].record_idx))
    for vector_index, item in enumerate(selected):
        item["vector_index"] = vector_index
    return selected


def item_used(item: dict[str, Any], variant: str) -> bool:
    dataset_cap, record_cap = variant_spec(variant)
    return (
        (dataset_cap is None or int(item["dataset_rank"]) < dataset_cap)
        and int(item["record_rank"]) < record_cap
    )


def common_family_eligibility(
    source_best: dict[str, base.RecordRef],
    cache_items: list[dict[str, Any]],
) -> dict[str, Any]:
    available_datasets = {
        item["source_key"]: int(item["available_distinct_datasets"])
        for item in cache_items
    }
    family_role_sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source_key, ref in source_best.items():
        family_role_sources[(ref.model_name, ref.role)].append(source_key)
    eligible: dict[str, list[str]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for (model_name, role), source_keys in sorted(family_role_sources.items()):
        undercovered = sorted(
            source_key for source_key in source_keys if available_datasets[source_key] < 2
        )
        if undercovered:
            excluded.append(
                {
                    "model_name": model_name,
                    "role": role,
                    "undercovered_source_keys": undercovered,
                    "rule": "common exclusion for every density variant: every native source needs >=2 datasets",
                }
            )
        else:
            eligible[role].append(model_name)
    result = {
        "minimum_distinct_datasets_per_native_source": 2,
        "eligible_family_roles": {
            role: sorted(eligible[role]) for role in base.ROLES
        },
        "eligible_family_count_per_role": {
            role: len(eligible[role]) for role in base.ROLES
        },
        "excluded_family_roles": excluded,
        "same_cohort_for_all_variants": True,
    }
    if any(result["eligible_family_count_per_role"][role] < 8 for role in base.ROLES):
        raise RuntimeError(f"common family support below 8: {result}")
    return result


def build_heldout_panel(
    refs_by_source: dict[str, list[base.RecordRef]],
    best_by_source: dict[str, base.RecordRef],
    *,
    seed: int,
) -> list[dict[str, base.RecordRef]]:
    heldout_core = [
        ref for ref in best_by_source.values()
        if tuple(ref.weight_shape) == base.ROLE_SHAPES[ref.role]
    ]
    if len(heldout_core) != 72:
        raise RuntimeError(f"expected 72 exact heldout matrices, got {len(heldout_core)}")
    panel: list[dict[str, base.RecordRef]] = []
    for ref in sorted(heldout_core, key=lambda value: (value.depth, base.ROLES.index(value.role))):
        refs = refs_by_source[ref.source_key]
        panel_a = [
            value for value in refs
            if int(base.ref_hash(value, seed, "heldout_panel"), 16) % 2 == 0
        ]
        panel_b = [
            value for value in refs
            if int(base.ref_hash(value, seed, "heldout_panel"), 16) % 2 == 1
        ]
        if not panel_a or not panel_b:
            ordered = sorted(refs, key=lambda value: base.ref_hash(value, seed, "heldout_fallback"))
            panel_a, panel_b = ordered[::2], ordered[1::2]
        context = min(panel_a, key=lambda value: base.ref_hash(value, seed, "context_a"))
        score = min(panel_b, key=lambda value: base.ref_hash(value, seed, "score_b"))
        if (context.chunk_idx, context.record_idx) == (score.chunk_idx, score.record_idx):
            raise RuntimeError(f"heldout A/B refs coincide: {ref.source_key}")
        panel.append({"context": context, "score": score})
    return panel


def sparse_reference_lock(
    reference_dir: Path,
    cache_items: list[dict[str, Any]],
    gain_refs: list[base.RecordRef],
    heldout_panel: list[dict[str, base.RecordRef]],
) -> dict[str, Any]:
    required = (
        "c0_sampling_manifest.json",
        "gain_sampling_manifest.json",
        "heldout_panel_manifest.json",
        "condition_template_summary.csv",
        "matrix_metrics.csv",
    )
    missing = [name for name in required if not (reference_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"sparse reference incomplete: {reference_dir}/{missing}")
    old_c0 = load_json(reference_dir / "c0_sampling_manifest.json")
    old_c0_ids = {
        f"{item['ref']['source_key']}:{item['ref']['chunk_idx']}:{item['ref']['record_idx']}"
        for item in old_c0
    }
    new_c0_ids = {
        record_identity(item["ref"]) for item in cache_items if item_used(item, "d2_k1")
    }
    old_gain = load_json(reference_dir / "gain_sampling_manifest.json")
    new_gain = [ref.as_dict() for ref in gain_refs]
    old_heldout = load_json(reference_dir / "heldout_panel_manifest.json")
    new_heldout = [
        {"context_a": item["context"].as_dict(), "score_b": item["score"].as_dict()}
        for item in heldout_panel
    ]
    result = {
        "reference_dir": str(reference_dir.resolve()),
        "old_d2_k1_records": len(old_c0_ids),
        "new_d2_k1_records": len(new_c0_ids),
        "d2_k1_ref_set_exact": old_c0_ids == new_c0_ids,
        "d2_k1_ref_symmetric_difference": len(old_c0_ids ^ new_c0_ids),
        "gain_manifest_exact": old_gain == new_gain,
        "heldout_manifest_exact": old_heldout == new_heldout,
        "reference_template_summary_sha256": base.sha256_file(
            reference_dir / "condition_template_summary.csv"
        ),
        "reference_matrix_metrics_sha256": base.sha256_file(reference_dir / "matrix_metrics.csv"),
    }
    if not all(
        result[key]
        for key in ("d2_k1_ref_set_exact", "gain_manifest_exact", "heldout_manifest_exact")
    ):
        raise RuntimeError(f"old sparse contract reproduction failed before forward: {result}")
    return result


def load_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device,
    no_amp: bool,
) -> tuple[torch.nn.Module, bool, torch.dtype | None, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = base.OmegaConf.create(payload["config"])
    model_cfg = base._build_model_cfg(cfg)
    model_cfg.big_vae.use_latent_sampling = False
    model_cfg.big_vae.rope_2d_coord_kind = "raw"
    model = base.build_weight_quantile_vae(model_cfg).to(device)
    model.load_state_dict(base._normalize_model_state_dict_keys(payload["model_state"]), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    amp_enabled, amp_dtype = base._resolve_amp(cfg, device)
    amp_enabled = bool(amp_enabled and not no_amp)
    info = {
        "path": str(checkpoint.resolve()),
        "sha256": base.sha256_file(checkpoint),
        "step": int(payload.get("step", 0)),
        "stage": int(payload.get("stage", 0)),
        "rope_2d_coord_kind": str(model.cfg.big_vae.rope_2d_coord_kind),
        "use_latent_sampling": bool(model.cfg.big_vae.use_latent_sampling),
        "strict_load": True,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
    }
    if (
        info["sha256"] != base.EXPECTED_CHECKPOINT_SHA256
        or info["step"] != base.EXPECTED_CHECKPOINT_STEP
        or info["rope_2d_coord_kind"] != "raw"
        or info["use_latent_sampling"] is not False
    ):
        raise RuntimeError(f"canonical checkpoint contract failed: {info}")
    return model, amp_enabled, amp_dtype, info


def cache_record_manifest(item: dict[str, Any]) -> dict[str, Any]:
    ref = item["ref"]
    return {
        "vector_index": item["vector_index"],
        "source_key": item["source_key"],
        "model_name": item["model_name"],
        "role": item["role"],
        "native_ordinal": item["native_ordinal"],
        "native_layers": item["native_layers"],
        "native_u": item["native_u"],
        "dataset": item["dataset"],
        "dataset_rank": item["dataset_rank"],
        "available_distinct_datasets": item["available_distinct_datasets"],
        "record_rank": item["record_rank"],
        "available_records_for_pair": item["available_records_for_pair"],
        "ref": ref.as_dict(),
        **{f"used_{variant}": item_used(item, variant) for variant in VARIANTS},
    }


def validate_cache_shard(
    tensor_path: Path,
    meta_path: Path,
    *,
    expected_contract: str,
    expected_items: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    meta = load_json(meta_path)
    expected_ids = [record_identity(item["ref"]) for item in expected_items]
    if (
        meta.get("cache_contract_sha256") != expected_contract
        or meta.get("record_ids") != expected_ids
        or meta.get("tensor_file_sha256") != base.sha256_file(tensor_path)
    ):
        raise RuntimeError(f"stale or mismatched cache shard: {tensor_path}")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    c_var = payload["c_var"].to(torch.float32).contiguous()
    c_patch = payload["c_patch"].to(torch.float32).contiguous()
    indices = payload["vector_indices"].tolist()
    expected_indices = [int(item["vector_index"]) for item in expected_items]
    if (
        indices != expected_indices
        or tuple(c_var.shape) != (len(expected_items), 256)
        or tuple(c_patch.shape) != (len(expected_items), 256)
        or not torch.isfinite(c_var).all()
        or not torch.isfinite(c_patch).all()
        or meta.get("c_var_sha256") != base.tensor_sha256(c_var)
        or meta.get("c_patch_sha256") != base.tensor_sha256(c_patch)
    ):
        raise RuntimeError(f"cache tensor validation failed: {tensor_path}")
    return c_var, c_patch


def extract_or_resume_cache(
    *,
    model: torch.nn.Module,
    source_dataset: base.OfflineBigVAEDataset,
    cache_items: list[dict[str, Any]],
    cache_dir: Path,
    cache_contract: str,
    shard_records: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    shard_total = math.ceil(len(cache_items) / shard_records)
    extracted = 0
    reused = 0
    started = time.monotonic()
    for shard_idx, begin in enumerate(range(0, len(cache_items), shard_records)):
        items = cache_items[begin : begin + shard_records]
        tensor_path = cache_dir / f"condition_vectors_{shard_idx:05d}.pt"
        meta_path = cache_dir / f"condition_vectors_{shard_idx:05d}.json"
        if tensor_path.is_file() and meta_path.is_file():
            c_var, c_patch = validate_cache_shard(
                tensor_path,
                meta_path,
                expected_contract=cache_contract,
                expected_items=items,
            )
            reused += len(items)
            logger.info(
                "stage=condition_cache hit shard=%s/%s records=%s path=%s",
                shard_idx + 1,
                shard_total,
                len(items),
                tensor_path,
            )
        elif tensor_path.exists() or meta_path.exists():
            # A crash between the atomic tensor rename and sidecar rename can
            # leave exactly half a shard.  It is not trusted or silently
            # deleted: quarantine the orphan(s), then recompute only this exact
            # manifest slice.  Fully present but mismatched shards still hard
            # fail in validate_cache_shard above.
            quarantine = cache_dir / "orphaned_partial_shards"
            quarantine.mkdir(parents=True, exist_ok=True)
            for orphan in (tensor_path, meta_path):
                if orphan.exists():
                    suffix = base.sha256_file(orphan)[:16]
                    destination = quarantine / f"{orphan.name}.{suffix}.orphaned"
                    if destination.exists():
                        raise RuntimeError(f"orphan quarantine collision: {destination}")
                    orphan.replace(destination)
                    logger.warning(
                        "stage=condition_cache quarantined_partial source=%s destination=%s",
                        orphan,
                        destination,
                    )
            shard_var = []
            shard_patch = []
            for local_idx, item in enumerate(items):
                _W, X = base.load_sample(source_dataset, item["ref"])
                c_var_one, c_patch_one, row_groups = base.extract_condition_vector(
                    model=model,
                    X=X,
                    source_key=item["source_key"],
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                shard_var.append(c_var_one)
                shard_patch.append(c_patch_one)
                extracted += 1
                if local_idx == 0 or (local_idx + 1) % 16 == 0 or local_idx + 1 == len(items):
                    logger.info(
                        "stage=condition_cache recompute_partial shard=%s/%s record=%s/%s row_groups=%s",
                        shard_idx + 1,
                        shard_total,
                        local_idx + 1,
                        len(items),
                        row_groups,
                    )
            c_var = torch.stack(shard_var).to(torch.float32).contiguous()
            c_patch = torch.stack(shard_patch).to(torch.float32).contiguous()
            payload = {
                "vector_indices": torch.tensor(
                    [int(item["vector_index"]) for item in items], dtype=torch.int64
                ),
                "c_var": c_var,
                "c_patch": c_patch,
            }
            temporary = tensor_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(tensor_path)
            atomic_json(
                meta_path,
                {
                    "cache_contract_sha256": cache_contract,
                    "record_ids": [record_identity(item["ref"]) for item in items],
                    "vector_indices": [int(item["vector_index"]) for item in items],
                    "tensor_file_sha256": base.sha256_file(tensor_path),
                    "c_var_sha256": base.tensor_sha256(c_var),
                    "c_patch_sha256": base.tensor_sha256(c_patch),
                    "dtype": "torch.float32",
                    "shape": [len(items), 256],
                    "recomputed_after_partial_quarantine": True,
                },
            )
        else:
            shard_var: list[torch.Tensor] = []
            shard_patch: list[torch.Tensor] = []
            for local_idx, item in enumerate(items):
                _W, X = base.load_sample(source_dataset, item["ref"])
                c_var_one, c_patch_one, row_groups = base.extract_condition_vector(
                    model=model,
                    X=X,
                    source_key=item["source_key"],
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                shard_var.append(c_var_one)
                shard_patch.append(c_patch_one)
                extracted += 1
                done = begin + local_idx + 1
                if local_idx == 0 or (local_idx + 1) % 16 == 0 or local_idx + 1 == len(items):
                    elapsed = time.monotonic() - started
                    logger.info(
                        "stage=condition_cache miss shard=%s/%s record=%s/%s model=%s role=%s "
                        "dataset=%s rank=%s row_groups=%s extracted_rate=%.2f/s elapsed=%.1fs",
                        shard_idx + 1,
                        shard_total,
                        done,
                        len(cache_items),
                        item["model_name"],
                        item["role"],
                        item["dataset"],
                        item["record_rank"],
                        row_groups,
                        extracted / max(elapsed, 1e-6),
                        elapsed,
                    )
            c_var = torch.stack(shard_var).to(torch.float32).contiguous()
            c_patch = torch.stack(shard_patch).to(torch.float32).contiguous()
            payload = {
                "vector_indices": torch.tensor(
                    [int(item["vector_index"]) for item in items], dtype=torch.int64
                ),
                "c_var": c_var,
                "c_patch": c_patch,
            }
            temporary = tensor_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(tensor_path)
            atomic_json(
                meta_path,
                {
                    "cache_contract_sha256": cache_contract,
                    "record_ids": [record_identity(item["ref"]) for item in items],
                    "vector_indices": [int(item["vector_index"]) for item in items],
                    "tensor_file_sha256": base.sha256_file(tensor_path),
                    "c_var_sha256": base.tensor_sha256(c_var),
                    "c_patch_sha256": base.tensor_sha256(c_patch),
                    "dtype": "torch.float32",
                    "shape": [len(items), 256],
                },
            )
        for item, c_var_one, c_patch_one in zip(items, c_var, c_patch, strict=True):
            vectors[int(item["vector_index"])] = (c_var_one.clone(), c_patch_one.clone())
    if len(vectors) != len(cache_items):
        raise RuntimeError(f"condition cache incomplete: {len(vectors)}/{len(cache_items)}")
    logger.info(
        "stage=condition_cache complete records=%s extracted=%s reused=%s elapsed=%.1fs",
        len(vectors),
        extracted,
        reused,
        time.monotonic() - started,
    )
    return vectors


def build_variant_template(
    *,
    variant: str,
    cache_items: list[dict[str, Any]],
    vectors: dict[int, tuple[torch.Tensor, torch.Tensor]],
    eligibility: dict[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, torch.Tensor]], list[dict[str, Any]]]:
    # Explicit hierarchy: equal records -> equal datasets -> source unit ->
    # interpolation within family -> equal families in role x canonical depth.
    by_source_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in cache_items:
        if item_used(item, variant):
            by_source_dataset[(item["source_key"], item["dataset"])].append(item)
    dataset_units: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    hierarchy_rows: list[dict[str, Any]] = []
    for key, items in sorted(by_source_dataset.items()):
        items.sort(key=lambda item: int(item["record_rank"]))
        dataset_units[key] = {
            "c_var": torch.stack([vectors[item["vector_index"]][0] for item in items]).mean(0),
            "c_patch": torch.stack([vectors[item["vector_index"]][1] for item in items]).mean(0),
        }
    by_source: dict[str, list[tuple[str, dict[str, torch.Tensor]]]] = defaultdict(list)
    for (source_key, dataset), value in dataset_units.items():
        by_source[source_key].append((dataset, value))
    source_units: list[dict[str, Any]] = []
    metadata_by_source = {item["source_key"]: item for item in cache_items}
    for source_key, values in sorted(by_source.items()):
        values.sort(key=lambda pair: next(
            int(item["dataset_rank"])
            for item in cache_items
            if item["source_key"] == source_key and item["dataset"] == pair[0]
        ))
        first = metadata_by_source[source_key]
        source_units.append(
            {
                "source_key": source_key,
                "model_name": first["model_name"],
                "role": first["role"],
                "native_ordinal": first["native_ordinal"],
                "native_layers": first["native_layers"],
                "native_u": first["native_u"],
                "datasets": [dataset for dataset, _value in values],
                "record_counts": [len(by_source_dataset[(source_key, dataset)]) for dataset, _value in values],
                "c_var": torch.stack([value["c_var"] for _dataset, value in values]).mean(0),
                "c_patch": torch.stack([value["c_patch"] for _dataset, value in values]).mean(0),
            }
        )
        hierarchy_rows.append(
            {
                "variant": variant,
                "source_key": source_key,
                "model_name": first["model_name"],
                "role": first["role"],
                "native_ordinal": first["native_ordinal"],
                "native_layers": first["native_layers"],
                "datasets": [dataset for dataset, _value in values],
                "records_per_dataset": [
                    len(by_source_dataset[(source_key, dataset)]) for dataset, _value in values
                ],
            }
        )

    by_model_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in source_units:
        by_model_role[(unit["model_name"], unit["role"])].append(unit)
    grid_u = torch.linspace(0.0, 1.0, 12)
    family_curves: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    for (model_name, role), units in sorted(by_model_role.items()):
        if model_name not in eligibility["eligible_family_roles"][role]:
            continue
        units.sort(key=lambda item: int(item["native_ordinal"]))
        layer_count = int(units[0]["native_layers"])
        if [int(item["native_ordinal"]) for item in units] != list(range(layer_count)):
            raise RuntimeError(f"incomplete source depth grid: {variant}/{model_name}/{role}")
        native_u = torch.tensor([float(item["native_u"]) for item in units])
        var_interp = base.interpolate_features(
            native_u, torch.stack([item["c_var"] for item in units]), grid_u
        )
        patch_interp = base.interpolate_features(
            native_u, torch.stack([item["c_patch"] for item in units]), grid_u
        )
        for depth in range(12):
            family_curves[(model_name, role, depth)] = {
                "c_var": var_interp[depth],
                "c_patch": patch_interp[depth],
            }
    cells: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for role in base.ROLES:
        families = eligibility["eligible_family_roles"][role]
        for depth in range(12):
            cells[(role, depth)] = {
                "c_var": torch.stack(
                    [family_curves[(family, role, depth)]["c_var"] for family in families]
                ).mean(0),
                "c_patch": torch.stack(
                    [family_curves[(family, role, depth)]["c_patch"] for family in families]
                ).mean(0),
            }
    expected = {(role, depth) for role in base.ROLES for depth in range(12)}
    if set(cells) != expected:
        raise RuntimeError(f"template grid incomplete: {variant}")
    return cells, hierarchy_rows


def condition_variance_components(
    cache_items: list[dict[str, Any]],
    vectors: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> list[dict[str, Any]]:
    """Equal-dataset within/between dispersion on the shared all_k8 cache."""
    by_source_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in cache_items:
        by_source_dataset[(item["source_key"], item["dataset"])].append(item)
    by_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (source_key, dataset), items in by_source_dataset.items():
        items.sort(key=lambda item: int(item["record_rank"]))
        by_source[source_key].append((dataset, items))
    output: list[dict[str, Any]] = []
    for source_key, dataset_groups in sorted(by_source.items()):
        dataset_groups.sort(key=lambda pair: int(pair[1][0]["dataset_rank"]))
        first = dataset_groups[0][1][0]
        for vector_kind, vector_offset in (("c_var", 0), ("c_patch", 1)):
            dataset_means: list[torch.Tensor] = []
            within_by_dataset: list[torch.Tensor] = []
            for _dataset, items in dataset_groups:
                values = torch.stack(
                    [vectors[int(item["vector_index"])][vector_offset] for item in items]
                ).to(torch.float64)
                mean = values.mean(0)
                dataset_means.append(mean)
                within_by_dataset.append((values - mean).square().mean())
            means = torch.stack(dataset_means)
            source_mean = means.mean(0)
            within_mse = torch.stack(within_by_dataset).mean()
            between_mse = (means - source_mean).square().mean()
            output.append(
                {
                    "source_key": source_key,
                    "model_name": first["model_name"],
                    "role": first["role"],
                    "native_ordinal": first["native_ordinal"],
                    "native_layers": first["native_layers"],
                    "vector_kind": vector_kind,
                    "datasets": len(dataset_groups),
                    "records_total_up_to_8": sum(len(items) for _dataset, items in dataset_groups),
                    "records_per_dataset": json.dumps(
                        [len(items) for _dataset, items in dataset_groups]
                    ),
                    "equal_dataset_within_record_mse": float(within_mse),
                    "between_dataset_mean_mse": float(between_mse),
                    "within_positive": float(within_mse) > 0,
                    "between_over_within": (
                        float(between_mse / within_mse) if float(within_mse) > 0 else ""
                    ),
                }
            )
    return output


def compare_sparse_template_hashes(
    templates: dict[str, dict[tuple[str, int], dict[str, torch.Tensor]]],
    reference_dir: Path,
) -> dict[str, Any]:
    with (reference_dir / "condition_template_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (row["role"], int(row["canonical_depth"])): (
            row["c_var_sha256"], row["c_patch_sha256"]
        )
        for row in rows if row["kind"] == "cell_mean"
    }
    actual = {
        key: (base.tensor_sha256(value["c_var"]), base.tensor_sha256(value["c_patch"]))
        for key, value in templates["d2_k1"].items()
    }
    mismatches = [
        {"role": key[0], "canonical_depth": key[1], "expected": expected.get(key), "actual": actual.get(key)}
        for key in sorted(set(expected) | set(actual)) if expected.get(key) != actual.get(key)
    ]
    result = {
        "cells_expected": len(expected),
        "cells_actual": len(actual),
        "all_72_c_var_and_c_patch_hashes_exact": not mismatches and len(actual) == 72,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    if not result["all_72_c_var_and_c_patch_hashes_exact"]:
        raise RuntimeError(f"d2_k1 template does not reproduce locked sparse template: {result}")
    return result


@torch.inference_mode()
def decode_fixed_matrix(
    *,
    model: torch.nn.Module,
    record: base.MatrixRecord,
    tiling: base.Tiling,
    template: dict[str, torch.Tensor],
    method: str,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> torch.Tensor:
    tiles = base.split_tiles(record.W, tiling)
    predictions: list[torch.Tensor] = []
    started = time.monotonic()
    for batch_idx, begin in enumerate(range(0, int(tiles.shape[0]), batch_size)):
        end = min(begin + batch_size, int(tiles.shape[0]))
        with base._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            prediction = base.fixed_forward(model, tiles[begin:end].to(device), template)
        predictions.append(prediction.float().cpu())
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or end == int(tiles.shape[0]):
            logger.info(
                "stage=ae_decode method=%s seed=%s depth=%s role=%s tiles=%s/%s rate=%.1f/s elapsed=%.1fs",
                method,
                tiling.seed,
                record.depth,
                record.role,
                end,
                int(tiles.shape[0]),
                end / max(time.monotonic() - started, 1e-6),
                time.monotonic() - started,
            )
    prediction, coverage = base.reassemble(
        torch.cat(predictions), tiling, tuple(record.W.shape)
    )
    if not torch.all(coverage == 1) or not torch.isfinite(prediction).all():
        raise RuntimeError(f"invalid decoded matrix: {method}/{record.source_key}")
    return prediction


def aggregate_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sum_keys = (
        "w_target", "w_pred", "w_dot", "w_error",
        "x_target", "x_pred", "x_dot", "x_error",
        "w_error_scaled", "x_error_scaled",
    )
    for seed in sorted({int(row["tiling_seed"]) for row in matrix_rows}):
        for method in METHODS:
            group = [
                row for row in matrix_rows
                if int(row["tiling_seed"]) == seed and row["method"] == method
            ]
            if len(group) != 72:
                raise RuntimeError(f"aggregate grid incomplete: {seed}/{method}/{len(group)}")
            role_values: dict[str, dict[str, float]] = {}
            for role in base.ROLES:
                role_group = [row for row in group if row["role"] == role]
                sums = {key: sum(float(row[key]) for row in role_group) for key in sum_keys}
                values = {
                    "raw_E_W": sums["w_error"] / sums["w_target"],
                    "raw_E_X": sums["x_error"] / sums["x_target"],
                    "calibrated_E_W": sums["w_error_scaled"] / sums["w_target"],
                    "calibrated_E_X": sums["x_error_scaled"] / sums["x_target"],
                    "raw_weight_cosine": sums["w_dot"] / math.sqrt(max(sums["w_target"] * sums["w_pred"], 1e-300)),
                    "raw_operator_cosine": sums["x_dot"] / math.sqrt(max(sums["x_target"] * sums["x_pred"], 1e-300)),
                    "raw_weight_norm_ratio": math.sqrt(sums["w_pred"] / sums["w_target"]),
                    "raw_operator_norm_ratio": math.sqrt(sums["x_pred"] / sums["x_target"]),
                }
                role_values[role] = values
                output.append({"tiling_seed": seed, "method": method, "aggregation": role, **values})
            macro = {
                key: sum(role_values[role][key] for role in base.ROLES) / len(base.ROLES)
                for key in next(iter(role_values.values()))
            }
            output.append({"tiling_seed": seed, "method": method, "aggregation": "macro", **macro})
            sums = {key: sum(float(row[key]) for row in group) for key in sum_keys}
            micro = {
                "raw_E_W": sums["w_error"] / sums["w_target"],
                "raw_E_X": sums["x_error"] / sums["x_target"],
                "calibrated_E_W": sums["w_error_scaled"] / sums["w_target"],
                "calibrated_E_X": sums["x_error_scaled"] / sums["x_target"],
                "raw_weight_cosine": sums["w_dot"] / math.sqrt(max(sums["w_target"] * sums["w_pred"], 1e-300)),
                "raw_operator_cosine": sums["x_dot"] / math.sqrt(max(sums["x_target"] * sums["x_pred"], 1e-300)),
                "raw_weight_norm_ratio": math.sqrt(sums["w_pred"] / sums["w_target"]),
                "raw_operator_norm_ratio": math.sqrt(sums["x_pred"] / sums["x_target"]),
            }
            output.append({"tiling_seed": seed, "method": method, "aggregation": "micro", **micro})
    return output


def bootstrap_metric_samples(
    group: list[dict[str, Any]],
    draw_indices: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    by_depth_role = {(int(row["depth"]), str(row["role"])): row for row in group}
    if len(by_depth_role) != 72:
        raise RuntimeError("bootstrap matrix grid is not 12x6")
    role_arrays: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    keys = (
        "w_target", "w_pred", "w_dot", "w_error", "w_error_scaled",
        "x_target", "x_pred", "x_dot", "x_error", "x_error_scaled",
    )
    for role in base.ROLES:
        for key in keys:
            values = np.asarray([by_depth_role[(depth, role)][key] for depth in range(12)])
            role_arrays[role][key] = values[draw_indices].sum(axis=1)
    result: dict[tuple[str, str], np.ndarray] = {}
    for aggregation in ("macro", "micro"):
        per_role: dict[str, dict[str, np.ndarray]] = {}
        for role in base.ROLES:
            values = role_arrays[role]
            per_role[role] = {
                "raw_E_W": values["w_error"] / values["w_target"],
                "raw_E_X": values["x_error"] / values["x_target"],
                "calibrated_E_W": values["w_error_scaled"] / values["w_target"],
                "calibrated_E_X": values["x_error_scaled"] / values["x_target"],
                "raw_weight_cosine": values["w_dot"] / np.sqrt(
                    np.maximum(values["w_target"] * values["w_pred"], 1e-300)
                ),
                "raw_operator_cosine": values["x_dot"] / np.sqrt(
                    np.maximum(values["x_target"] * values["x_pred"], 1e-300)
                ),
            }
        if aggregation == "macro":
            for metric in next(iter(per_role.values())):
                result[(aggregation, metric)] = np.stack(
                    [per_role[role][metric] for role in base.ROLES], axis=1
                ).mean(axis=1)
        else:
            sums = {
                key: sum(role_arrays[role][key] for role in base.ROLES) for key in keys
            }
            result[(aggregation, "raw_E_W")] = sums["w_error"] / sums["w_target"]
            result[(aggregation, "raw_E_X")] = sums["x_error"] / sums["x_target"]
            result[(aggregation, "calibrated_E_W")] = sums["w_error_scaled"] / sums["w_target"]
            result[(aggregation, "calibrated_E_X")] = sums["x_error_scaled"] / sums["x_target"]
            result[(aggregation, "raw_weight_cosine")] = sums["w_dot"] / np.sqrt(
                np.maximum(sums["w_target"] * sums["w_pred"], 1e-300)
            )
            result[(aggregation, "raw_operator_cosine")] = sums["x_dot"] / np.sqrt(
                np.maximum(sums["x_target"] * sums["x_pred"], 1e-300)
            )
    return result


def paired_bootstrap(
    matrix_rows: list[dict[str, Any]], *, draws: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, 12, size=(draws, 12))
    absolute: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    comparisons = (
        ("d2_k4_minus_d2_k1", "d2_k4", "d2_k1", "within_dataset_K1_to_K4_at_D2"),
        ("all_k4_minus_all_k1", "all_k4", "all_k1", "within_dataset_K1_to_K4_at_allD"),
        ("all_k1_minus_d2_k1", "all_k1", "d2_k1", "dataset_breadth_at_K1"),
        ("all_k4_minus_d2_k4", "all_k4", "d2_k4", "dataset_breadth_at_K4"),
        ("all_k8_minus_d2_k8", "all_k8", "d2_k8", "dataset_breadth_at_K8"),
        ("all_k4_minus_d2_k1", "all_k4", "d2_k1", "predeclared_dense_primary_vs_sparse"),
        ("all_k8_minus_d2_k1", "all_k8", "d2_k1", "predeclared_max_density_primary_vs_sparse"),
        ("d2_k8_minus_d2_k4", "d2_k8", "d2_k4", "K4_to_K8_saturation_at_D2"),
        ("all_k8_minus_all_k4", "all_k8", "all_k4", "K4_to_K8_saturation_at_allD"),
    )
    for tiling_seed in sorted({int(row["tiling_seed"]) for row in matrix_rows}):
        samples: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        for method in METHODS:
            group = [
                row for row in matrix_rows
                if int(row["tiling_seed"]) == tiling_seed and row["method"] == method
            ]
            samples[method] = bootstrap_metric_samples(group, draw_indices)
            for (aggregation, metric), values in samples[method].items():
                absolute.append(
                    {
                        "tiling_seed": tiling_seed,
                        "method": method,
                        "aggregation": aggregation,
                        "metric": metric,
                        "draws": draws,
                        "mean": float(values.mean()),
                        "l95": float(np.quantile(values, 0.05)),
                        "u95": float(np.quantile(values, 0.95)),
                    }
                )
        for contrast_name, candidate, reference, interpretation in comparisons:
            for key in samples[candidate]:
                delta = samples[candidate][key] - samples[reference][key]
                aggregation, metric = key
                lower_is_better = metric in {
                    "raw_E_W", "raw_E_X", "calibrated_E_W", "calibrated_E_X"
                }
                contrasts.append(
                    {
                        "tiling_seed": tiling_seed,
                        "contrast": contrast_name,
                        "candidate": candidate,
                        "reference": reference,
                        "interpretation": interpretation,
                        "aggregation": aggregation,
                        "metric": metric,
                        "lower_is_better": lower_is_better,
                        "draws": draws,
                        "mean_delta": float(delta.mean()),
                        "l95_delta": float(np.quantile(delta, 0.05)),
                        "u95_delta": float(np.quantile(delta, 0.95)),
                        "probability_improves": float(
                            np.mean(delta < 0) if lower_is_better else np.mean(delta > 0)
                        ),
                    }
                )
        interactions = (
            (
                "breadth_x_K1_to_K4_interaction",
                ("all_k4", "all_k1"),
                ("d2_k4", "d2_k1"),
                "(all_k4-all_k1) - (d2_k4-d2_k1)",
            ),
            (
                "breadth_x_K4_to_K8_interaction",
                ("all_k8", "all_k4"),
                ("d2_k8", "d2_k4"),
                "(all_k8-all_k4) - (d2_k8-d2_k4)",
            ),
        )
        for contrast_name, (all_high, all_low), (d2_high, d2_low), formula in interactions:
            for key in samples[all_high]:
                delta = (
                    samples[all_high][key] - samples[all_low][key]
                    - samples[d2_high][key] + samples[d2_low][key]
                )
                aggregation, metric = key
                lower_is_better = metric in {
                    "raw_E_W", "raw_E_X", "calibrated_E_W", "calibrated_E_X"
                }
                contrasts.append(
                    {
                        "tiling_seed": tiling_seed,
                        "contrast": contrast_name,
                        "candidate": "difference_in_differences",
                        "reference": "zero_interaction",
                        "interpretation": formula,
                        "aggregation": aggregation,
                        "metric": metric,
                        "lower_is_better": lower_is_better,
                        "draws": draws,
                        "mean_delta": float(delta.mean()),
                        "l95_delta": float(np.quantile(delta, 0.05)),
                        "u95_delta": float(np.quantile(delta, 0.95)),
                        "probability_improves": "",
                    }
                )
    return absolute, contrasts


def plot_density(aggregate: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ("raw_E_X", "calibrated_E_X", "raw_operator_cosine")
    seeds = sorted({int(row["tiling_seed"]) for row in aggregate})
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(VARIANTS))
    for axis, metric in zip(axes, metrics, strict=True):
        for seed_idx, tiling_seed in enumerate(seeds):
            values = [
                next(
                    row for row in aggregate
                    if int(row["tiling_seed"]) == tiling_seed
                    and row["method"] == method
                    and row["aggregation"] == "macro"
                )[metric]
                for method in VARIANTS
            ]
            axis.plot(x, values, marker="o", label=str(tiling_seed), alpha=0.85)
        axis.set_xticks(x, VARIANTS, rotation=35, ha="right")
        axis.set_title(f"macro {metric}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].axhline(1.0, linestyle=":", color="black", linewidth=1)
    axes[-1].legend(title="tiling")
    figure.suptitle("Source-only activation-template density factorial")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Source-only nested condition-density gate")
    value.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    value.add_argument("--source-root", type=Path, default=base.DEFAULT_SOURCE_ROOT)
    value.add_argument("--heldout-root", type=Path, default=base.DEFAULT_HELDOUT_ROOT)
    value.add_argument("--sparse-reference-dir", type=Path, default=DEFAULT_SPARSE_REFERENCE)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--cache-shard-records", type=int, default=128)
    value.add_argument("--seed", type=int, default=260816)
    value.add_argument("--tiling-seeds", default="26081601,26081602")
    value.add_argument("--gain-matrices-per-role", type=int, default=12)
    value.add_argument("--bootstrap-draws", type=int, default=10000)
    value.add_argument("--no-amp", action="store_true")
    value.add_argument("--audit-only", action="store_true")
    value.add_argument("--cache-and-template-only", action="store_true")
    return value


def assert_target_seal(args: argparse.Namespace) -> dict[str, Any]:
    """Hard-stop if this source evaluator is pointed outside its frozen roots."""
    forbidden_module_fragments = ("data2vec", "wav2vec", "whisper", "llama")
    forbidden_loaded = sorted(
        name for name in sys.modules
        if any(fragment in name.lower() for fragment in forbidden_module_fragments)
    )
    assertions = {
        "source_root_is_exact_frozen_vision_source": (
            args.source_root.resolve() == base.DEFAULT_SOURCE_ROOT.resolve()
        ),
        "heldout_root_is_exact_frozen_vit_b_source_holdout": (
            args.heldout_root.resolve() == base.DEFAULT_HELDOUT_ROOT.resolve()
        ),
        "cli_exposes_no_target_domain_root": not any(
            action.dest.startswith("target") for action in parser()._actions
        ),
        "forbidden_target_model_modules_loaded": forbidden_loaded,
        "target_result_read_or_forward_authorized": False,
    }
    if (
        not assertions["source_root_is_exact_frozen_vision_source"]
        or not assertions["heldout_root_is_exact_frozen_vit_b_source_holdout"]
        or not assertions["cli_exposes_no_target_domain_root"]
        or forbidden_loaded
    ):
        raise RuntimeError(f"target seal assertion failed: {assertions}")
    return assertions


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", mode="w")],
        force=True,
    )
    logger = logging.getLogger("source_condition_density")
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tiling_seeds = tuple(int(value) for value in args.tiling_seeds.split(","))
    if len(tiling_seeds) != 2 or len(set(tiling_seeds)) != 2:
        raise ValueError("exactly two distinct tiling seeds are required")
    if args.cache_shard_records <= 0:
        raise ValueError("cache-shard-records must be positive")
    target_seal = assert_target_seal(args)

    resolved = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_root": str(args.source_root.resolve()),
        "heldout_root": str(args.heldout_root.resolve()),
        "sparse_reference_dir": str(args.sparse_reference_dir.resolve()),
        "output_dir": str(output),
        "device": str(device),
        "dtype": "canonical checkpoint AMP policy",
        "seed": args.seed,
        "tiling_seeds": list(tiling_seeds),
        "batch_size": args.batch_size,
        "cache_shard_records": args.cache_shard_records,
        "variants": [
            {
                "name": name,
                "dataset_cap": dataset_cap or "all available dataset identities",
                "record_cap": f"first up to {record_cap} hash-ranked records per dataset",
            }
            for name, dataset_cap, record_cap in VARIANT_SPECS
        ],
        "hierarchy": "equal records -> equal datasets -> source -> family curve -> equal families",
        "common_eligibility": "every native source in family x role has >=2 datasets; identical for all variants",
        "gain_matrices_per_role": args.gain_matrices_per_role,
        "bootstrap_draws": args.bootstrap_draws,
        "primary_density_contrast": "all_k8 minus d2_k1",
        "primary_inferential_endpoint": (
            "paired source-block bootstrap raw macro E_X; one-sided U95(delta)<0 on both tilings"
        ),
        "mechanistic_secondary_endpoints": [
            "paired calibrated macro E_X delta",
            "paired macro raw operator cosine delta",
            "micro and per-role heterogeneity",
        ],
        "k8_status": "predeclared maximum-density primary; all_k8-minus-all_k4 is saturation secondary",
        "cache_mode": "resumable hash-validated per-record C-vector shards",
        "all_dataset_semantics": (
            "all available dataset identities for a source, but only first up to K hash-ranked "
            "activation records within each dataset; never all 290,966 records"
        ),
        "partial_cache_policy": (
            "quarantine one-sided pt/json crash remnants and recompute only that exact locked shard; "
            "fully present mismatched shards hard-fail"
        ),
        "target_domain_access": False,
        "target_action_regardless_of_result": "REMAINS_SEALED; source diagnostic cannot post-hoc rescue old G1",
        "audit_only": bool(args.audit_only),
        "cache_and_template_only": bool(args.cache_and_template_only),
        "progress": "verbose",
    }
    base.write_json(output / "resolved_config.json", resolved)
    base.write_json(output / "target_seal_assertions.json", target_seal)
    logger.info("resolved_config=%s", json.dumps(resolved, sort_keys=True))

    logger.info("stage=index_audit")
    source_refs, source_best, source_audit = base.scan_index(
        args.source_root, keep_model=None, seed=args.seed, logger=logger
    )
    heldout_refs, heldout_best, heldout_audit = base.scan_index(
        args.heldout_root, keep_model="vit_base_p16_224", seed=args.seed, logger=logger
    )
    source_models = sorted({ref.model_name for ref in source_best.values()})
    if len(source_models) != 11 or "vit_base_p16_224" in source_models:
        raise RuntimeError(f"source split invalid: {source_models}")
    structure, structure_audit = base.build_structure(source_best.values())
    cache_items = select_cache_records(
        source_refs, source_best, structure=structure, seed=args.seed
    )
    if len(cache_items) != 12180:
        raise RuntimeError(f"locked all_k8 cache count changed: {len(cache_items)}")
    support_counts = {variant: sum(item_used(item, variant) for item in cache_items) for variant in VARIANTS}
    expected_counts = {
        "d2_k1": 1752,
        "d2_k4": 6049,
        "d2_k8": 11173,
        "all_k1": 2100,
        "all_k4": 6780,
        "all_k8": 12180,
    }
    if support_counts != expected_counts:
        raise RuntimeError(f"locked nested support changed: {support_counts}")
    eligibility = common_family_eligibility(source_best, cache_items)
    gain_refs = base.select_gain_refs(
        source_best.values(),
        structure=structure,
        seed=args.seed,
        per_role=args.gain_matrices_per_role,
    )
    heldout_panel = build_heldout_panel(heldout_refs, heldout_best, seed=args.seed)
    reference_lock = sparse_reference_lock(
        args.sparse_reference_dir, cache_items, gain_refs, heldout_panel
    )
    cache_manifest = [cache_record_manifest(item) for item in cache_items]
    base.write_json(output / "condition_vector_selection_manifest.json", cache_manifest)
    base.write_json(output / "gain_sampling_manifest.json", [ref.as_dict() for ref in gain_refs])
    base.write_json(
        output / "heldout_panel_manifest.json",
        [
            {"context_a": item["context"].as_dict(), "score_b": item["score"].as_dict()}
            for item in heldout_panel
        ],
    )
    base.write_json(output / "common_family_eligibility.json", eligibility)
    base.write_json(output / "sparse_reference_lock.json", reference_lock)
    index_audit = {
        "source": source_audit,
        "heldout_source": heldout_audit,
        "source_models": source_models,
        "source_structure": structure_audit,
        "cache_records_all_k8": len(cache_items),
        "nested_variant_record_counts": support_counts,
        "gain_matrices": len(gain_refs),
        "heldout_matrices": len(heldout_panel),
        "target_access": False,
    }
    base.write_json(output / "index_audit.json", index_audit)
    execution = {
        "status": "LOCKED_BEFORE_ANY_AE_OR_DISTRIBUTION_ENCODER_FORWARD",
        "implementation_sha256": base.sha256_file(Path(__file__).resolve()),
        "base_evaluator_sha256": base.sha256_file(Path(base.__file__).resolve()),
        "resolved_config_sha256": base.sha256_file(output / "resolved_config.json"),
        "selection_manifest_sha256": base.sha256_file(output / "condition_vector_selection_manifest.json"),
        "gain_manifest_sha256": base.sha256_file(output / "gain_sampling_manifest.json"),
        "heldout_manifest_sha256": base.sha256_file(output / "heldout_panel_manifest.json"),
        "eligibility_sha256": base.sha256_file(output / "common_family_eligibility.json"),
        "sparse_reference_lock_sha256": base.sha256_file(output / "sparse_reference_lock.json"),
        "target_seal_assertions_sha256": base.sha256_file(output / "target_seal_assertions.json"),
        "target_access": False,
    }
    base.write_json(output / "execution_manifest.json", execution)
    logger.info("stage=pre_forward_lock artifact=%s", output / "execution_manifest.json")
    if args.audit_only:
        logger.info("completed audit_only=true; no checkpoint/model/distribution forward")
        return

    logger.info("stage=checkpoint_load")
    model, amp_enabled, amp_dtype, checkpoint_info = load_checkpoint(
        args.checkpoint, device=device, no_amp=args.no_amp
    )
    base.write_json(output / "checkpoint_info.json", checkpoint_info)
    logger.info("checkpoint_info=%s", checkpoint_info)
    cache_contract = base.stable_hex(
        "source_condition_vector_cache_v1",
        checkpoint_info["sha256"],
        checkpoint_info["step"],
        checkpoint_info["rope_2d_coord_kind"],
        checkpoint_info["amp_enabled"],
        checkpoint_info["amp_dtype"],
        execution["selection_manifest_sha256"],
        execution["base_evaluator_sha256"],
    )
    source_dataset = base.build_dataset(args.source_root, args.seed)
    vectors = extract_or_resume_cache(
        model=model,
        source_dataset=source_dataset,
        cache_items=cache_items,
        cache_dir=output / "condition_vector_cache",
        cache_contract=cache_contract,
        shard_records=args.cache_shard_records,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        logger=logger,
    )
    variance_rows = condition_variance_components(cache_items, vectors)
    base.write_csv(output / "condition_record_dataset_variance.csv", variance_rows)
    templates: dict[str, dict[tuple[str, int], dict[str, torch.Tensor]]] = {}
    hierarchy_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        logger.info("stage=template_build variant=%s", variant)
        templates[variant], variant_hierarchy = build_variant_template(
            variant=variant,
            cache_items=cache_items,
            vectors=vectors,
            eligibility=eligibility,
        )
        hierarchy_rows.extend(variant_hierarchy)
        for (role, depth), value in templates[variant].items():
            template_rows.append(
                {
                    "variant": variant,
                    "role": role,
                    "canonical_depth": depth,
                    "c_var_sha256": base.tensor_sha256(value["c_var"]),
                    "c_patch_sha256": base.tensor_sha256(value["c_patch"]),
                    "c_var_norm": float(value["c_var"].norm()),
                    "c_patch_norm": float(value["c_patch"].norm()),
                }
            )
    sparse_template_lock = compare_sparse_template_hashes(templates, args.sparse_reference_dir)
    base.write_json(output / "sparse_template_reproduction.json", sparse_template_lock)
    torch.save(templates, output / "source_condition_density_templates.pt")
    base.write_csv(output / "condition_template_summary.csv", template_rows)
    base.write_json(output / "condition_hierarchy_manifest.json", hierarchy_rows)
    execution = load_json(output / "execution_manifest.json")
    execution["template_cache_lock"] = {
        "status": "LOCKED_DENSE_TEMPLATES_BEFORE_FIRST_WEIGHT_AE_FORWARD",
        "cache_contract_sha256": cache_contract,
        "cache_shards": math.ceil(len(cache_items) / args.cache_shard_records),
        "template_tensor_cells": len(template_rows),
        "templates_file_sha256": base.sha256_file(output / "source_condition_density_templates.pt"),
        "template_summary_sha256": base.sha256_file(output / "condition_template_summary.csv"),
        "hierarchy_manifest_sha256": base.sha256_file(output / "condition_hierarchy_manifest.json"),
        "record_dataset_variance_sha256": base.sha256_file(output / "condition_record_dataset_variance.csv"),
        "sparse_template_reproduction_sha256": base.sha256_file(output / "sparse_template_reproduction.json"),
    }
    base.write_json(output / "execution_manifest.json", execution)
    if args.cache_and_template_only:
        logger.info("completed cache_and_template_only=true; no Weight-AE reconstruction forward")
        return

    logger.info("stage=heldout_load matrices=72")
    heldout_dataset = base.build_dataset(args.heldout_root, args.seed)
    heldout_records: list[base.MatrixRecord] = []
    for index, item in enumerate(heldout_panel):
        W_a, X_a = base.load_sample(heldout_dataset, item["context"])
        W_b, X_b = base.load_sample(heldout_dataset, item["score"])
        if not torch.equal(W_a, W_b) or torch.equal(X_a, X_b):
            raise RuntimeError(f"heldout A/B contract failed: {item}")
        ref = item["context"]
        heldout_records.append(
            base.MatrixRecord(
                model_name=ref.model_name,
                depth=ref.depth,
                canonical_depth=ref.depth,
                role=ref.role,
                layer_name=ref.layer_name,
                source_key=ref.source_key,
                context_ref=ref,
                score_ref=item["score"],
                W=W_a,
                X_context=X_a,
                X_score=X_b,
            )
        )
        logger.info("stage=heldout_load matrix=%s/72 depth=%s role=%s", index + 1, ref.depth, ref.role)
    if {(record.depth, record.role) for record in heldout_records} != {
        (depth, role) for depth in range(12) for role in base.ROLES
    }:
        raise RuntimeError("heldout grid is not exact 12x6")
    coverage_rows = [
        base.validate_tiling(record, base.make_tiling(record, tiling_seed))
        for tiling_seed in tiling_seeds for record in heldout_records
    ]
    base.write_csv(output / "tiling_coverage.csv", coverage_rows)
    for tiling_seed in tiling_seeds:
        tile_count = sum(
            int(row["num_tiles"]) for row in coverage_rows
            if int(row["tiling_seed"]) == tiling_seed
        )
        if tile_count != 20736:
            raise RuntimeError(f"tiling tile count changed: {tiling_seed}/{tile_count}")

    logger.info("stage=source_gain_panel_load matrices=%s", len(gain_refs))
    gain_records: list[base.MatrixRecord] = []
    for index, ref in enumerate(gain_refs):
        W, X = base.load_sample(source_dataset, ref)
        gain_records.append(
            base.MatrixRecord(
                model_name=ref.model_name,
                depth=ref.depth,
                canonical_depth=base.canonical_depth(*structure[ref.source_key]),
                role=ref.role,
                layer_name=ref.layer_name,
                source_key=ref.source_key,
                context_ref=ref,
                score_ref=ref,
                W=W,
                X_context=X,
                X_score=X,
            )
        )
        logger.info("stage=source_gain_panel_load matrix=%s/%s", index + 1, len(gain_refs))
    gain_seed = args.seed * 100 + 3
    gain_stat_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(gain_records):
        tiling = base.make_tiling(record, gain_seed)
        base.validate_tiling(record, tiling)
        for variant in VARIANTS:
            prediction = decode_fixed_matrix(
                model=model,
                record=record,
                tiling=tiling,
                template=templates[variant][(record.role, record.canonical_depth)],
                method=variant,
                device=device,
                batch_size=args.batch_size,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                logger=logger,
            )
            gain_stat_rows.append(
                {
                    "method": variant,
                    "role": record.role,
                    "model_name": record.model_name,
                    "depth": record.depth,
                    "canonical_depth": record.canonical_depth,
                    "source_key": record.source_key,
                    "tiling_seed": gain_seed,
                    "num_tiles": tiling.num_tiles,
                    "prediction_sha256": base.tensor_sha256(prediction),
                    **base.sufficient_stats(record.W, prediction, record.X_score),
                }
            )
        logger.info("stage=gain_fit_prediction matrix=%s/%s", record_index + 1, len(gain_records))
    base.write_csv(output / "source_gain_sufficient_stats.csv", gain_stat_rows)
    gains: dict[tuple[str, str], float] = {}
    gain_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for role in base.ROLES:
            group = [
                row for row in gain_stat_rows
                if row["method"] == variant and row["role"] == role
            ]
            gain_denominator = sum(float(row["x_pred"]) for row in group)
            if gain_denominator <= 0:
                raise RuntimeError(f"nonpositive source gain denominator: {variant}/{role}")
            gain = sum(float(row["x_dot"]) for row in group) / gain_denominator
            gains[(variant, role)] = gain
            gain_rows.append(
                {
                    "method": variant,
                    "role": role,
                    "gain": gain,
                    "in_allowed_interval": 0.25 <= gain <= 4.0,
                    "fit_matrices": len(group),
                }
            )
    base.write_csv(output / "source_fit_role_gains.csv", gain_rows)
    gain_validity = {
        "calibrated_secondary_valid": all(bool(row["in_allowed_interval"]) for row in gain_rows),
        "required_interval": [0.25, 4.0],
        "invalid_method_roles": [
            {"method": row["method"], "role": row["role"], "gain": row["gain"]}
            for row in gain_rows if not bool(row["in_allowed_interval"])
        ],
        "raw_primary_depends_on_fitted_gain": False,
        "policy": (
            "out-of-range gain invalidates calibrated secondary interpretation only; "
            "raw primary remains mathematically unchanged"
        ),
    }
    base.write_json(output / "source_gain_validity.json", gain_validity)

    logger.info("stage=heldout_evaluation")
    matrix_rows: list[dict[str, Any]] = []
    dummy_base_templates = {"global": next(iter(templates["d2_k1"].values()))}
    for tiling_seed in tiling_seeds:
        for record_index, record in enumerate(heldout_records):
            tiling = base.make_tiling(record, tiling_seed)
            predictions: dict[str, torch.Tensor] = {}
            for variant in VARIANTS:
                predictions[variant] = decode_fixed_matrix(
                    model=model,
                    record=record,
                    tiling=tiling,
                    template=templates[variant][(record.role, record.canonical_depth)],
                    method=variant,
                    device=device,
                    batch_size=args.batch_size,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    logger=logger,
                )
            for control in ("ae_native_c", "ae_zero_c"):
                predictions[control] = base.decode_matrix(
                    model=model,
                    record=record,
                    tiling=tiling,
                    method=control,
                    templates=dummy_base_templates,
                    device=device,
                    batch_size=args.batch_size,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    logger=logger,
                )
            predictions["identity"] = record.W.clone()
            predictions["zero"] = torch.zeros_like(record.W)
            for method in METHODS:
                gain = gains[(method, record.role)] if method in VARIANTS else 1.0
                row = base.metric_row(
                    record,
                    tiling_seed=tiling_seed,
                    method=method,
                    prediction=predictions[method],
                    gain=gain,
                    num_tiles=tiling.num_tiles,
                )
                if method == "zero":
                    row["raw_weight_cosine"] = 0.0
                    row["raw_operator_cosine"] = 0.0
                matrix_rows.append(row)
            logger.info(
                "stage=heldout_evaluation seed=%s matrix=%s/72 depth=%s role=%s",
                tiling_seed,
                record_index + 1,
                record.depth,
                record.role,
            )
    aggregate = aggregate_rows(matrix_rows)
    absolute_bootstrap, paired_contrasts = paired_bootstrap(
        matrix_rows, draws=args.bootstrap_draws, seed=args.seed + 99
    )
    base.write_csv(output / "matrix_metrics.csv", matrix_rows)
    base.write_csv(output / "aggregate_metrics.csv", aggregate)
    base.write_csv(output / "block_bootstrap_absolute.csv", absolute_bootstrap)
    base.write_csv(output / "block_bootstrap_paired_contrasts.csv", paired_contrasts)
    plot_density(aggregate, output / "condition_density_factorial.png")

    old_rows: list[dict[str, str]]
    with (args.sparse_reference_dir / "matrix_metrics.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    old_hashes = {
        (int(row["tiling_seed"]), int(row["depth"]), row["role"]): row["prediction_sha256"]
        for row in old_rows if row["method"] == "ae_cell_mean_c0"
    }
    new_hashes = {
        (int(row["tiling_seed"]), int(row["depth"]), row["role"]): row["prediction_sha256"]
        for row in matrix_rows if row["method"] == "d2_k1"
    }
    prediction_reproduction = {
        "expected_rows": len(old_hashes),
        "actual_rows": len(new_hashes),
        "all_prediction_hashes_exact": old_hashes == new_hashes,
        "mismatch_count": sum(
            old_hashes.get(key) != new_hashes.get(key)
            for key in set(old_hashes) | set(new_hashes)
        ),
    }
    base.write_json(output / "sparse_prediction_reproduction.json", prediction_reproduction)

    def contrast_row(tiling_seed: int, aggregation: str, metric: str) -> dict[str, Any]:
        return next(
            row for row in paired_contrasts
            if int(row["tiling_seed"]) == tiling_seed
            and row["contrast"] == "all_k8_minus_d2_k1"
            and row["aggregation"] == aggregation
            and row["metric"] == metric
        )

    zero_controls = all(
        0.999999 <= next(
            row for row in aggregate
            if int(row["tiling_seed"]) == tiling_seed
            and row["method"] == "zero"
            and row["aggregation"] == aggregation
        )[metric] <= 1.000001
        for tiling_seed in tiling_seeds
        for aggregation in ("macro", "micro")
        for metric in ("raw_E_W", "raw_E_X")
    )
    identity_controls = all(
        next(
            row for row in aggregate
            if int(row["tiling_seed"]) == tiling_seed
            and row["method"] == "identity"
            and row["aggregation"] == aggregation
        )[metric] <= 1e-10
        for tiling_seed in tiling_seeds
        for aggregation in ("macro", "micro")
        for metric in ("raw_E_W", "raw_E_X")
    )
    native_controls = all(
        next(
            row for row in aggregate
            if int(row["tiling_seed"]) == tiling_seed
            and row["method"] == "ae_native_c"
            and row["aggregation"] == aggregation
        )["raw_E_X"] < 1.0
        for tiling_seed in tiling_seeds
        for aggregation in ("macro", "micro")
    )
    g0 = bool(
        prediction_reproduction["all_prediction_hashes_exact"]
        and zero_controls
        and identity_controls
        and native_controls
    )
    density_checks: dict[str, Any] = {}
    for tiling_seed in tiling_seeds:
        raw_error = contrast_row(tiling_seed, "macro", "raw_E_X")
        calibrated_error = contrast_row(tiling_seed, "macro", "calibrated_E_X")
        direction = contrast_row(tiling_seed, "macro", "raw_operator_cosine")
        density_checks[str(tiling_seed)] = {
            "primary_paired_raw_macro_E_X_u95_delta_below_0": raw_error["u95_delta"] < 0,
            "secondary_paired_calibrated_macro_E_X_u95_delta_below_0": calibrated_error["u95_delta"] < 0,
            "secondary_paired_macro_operator_cosine_l95_delta_above_0": direction["l95_delta"] > 0,
            "raw_macro_E_X_delta": raw_error,
            "calibrated_macro_E_X_delta": calibrated_error,
            "macro_operator_cosine_delta": direction,
        }
    density_supported = bool(g0) and all(
        checks["primary_paired_raw_macro_E_X_u95_delta_below_0"]
        for checks in density_checks.values()
    )
    decision = {
        "G0_sparse_reproduction_and_controls": bool(g0),
        "G0_checks": {
            "sparse_prediction_hashes_exact": prediction_reproduction["all_prediction_hashes_exact"],
            "identity_controls": identity_controls,
            "zero_controls": zero_controls,
            "native_macro_micro_raw_E_X_below_zero": native_controls,
            "tiling_coverage_exact": True,
        },
        "source_density_effect_supported": density_supported,
        "primary_contrast": "all_k8 minus d2_k1",
        "primary_endpoint": "paired raw macro E_X delta; lower is better",
        "primary_decision_rule": "one-sided block-bootstrap U95(delta)<0 on both locked tilings",
        "secondary_endpoints_are_mechanistic_not_gate_vetoes": True,
        "calibrated_secondary_valid": gain_validity["calibrated_secondary_valid"],
        "calibrated_secondary_invalid_reason_if_any": gain_validity["invalid_method_roles"],
        "raw_primary_depends_on_fitted_gain": False,
        "primary_checks": density_checks,
        "interpretation": (
            "SOURCE_ONLY_DENSITY_EFFECT_SUPPORTED"
            if density_supported else "SOURCE_ONLY_DENSITY_EFFECT_NOT_ESTABLISHED"
        ),
        "target_action": "REMAINS_SEALED_REGARDLESS; DO_NOT_TREAT_THIS_POSTHOC_DIAGNOSTIC_AS_OLD_G1_RESCUE",
        "saturation_secondary": "all_k8 minus all_k4; see paired contrast CSV",
    }
    base.write_json(output / "density_decision.json", decision)
    manifest = {
        "status": "COMPLETE_SOURCE_CONDITION_DENSITY",
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": checkpoint_info,
        "counts": {
            "cache_records": len(cache_items),
            "variants": len(VARIANTS),
            "gain_stat_rows": len(gain_stat_rows),
            "heldout_matrices": len(heldout_records),
            "matrix_metric_rows": len(matrix_rows),
            "tiles_per_method_per_tiling": 20736,
        },
        "decision": decision,
        "target_access": False,
        "artifacts": sorted(str(path) for path in output.iterdir()),
    }
    base.write_json(output / "run_manifest.json", manifest)
    logger.info(
        "completed elapsed_s=%.1f G0=%s density_supported=%s target_action=%s artifacts=%s",
        manifest["elapsed_seconds"],
        g0,
        density_supported,
        decision["target_action"],
        manifest["artifacts"],
    )


if __name__ == "__main__":
    main()
