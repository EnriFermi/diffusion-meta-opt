from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from source_confirmatory_g01_gate import RecordRef, parse_ref, ref_hash, stable_hex


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/training/checkpoints/"
    "weight_quantile_vae/stage_1/offline_dataset"
)
DEFAULT_OUTPUT = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_condition_density_design_20260816"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_order(seed: int, source_key: str, dataset: str) -> str:
    # Deliberately identical to source_confirmatory_g01_gate.select_c0_refs.
    return stable_hex(seed, "c0_dataset", source_key, dataset)


def record_order(seed: int, ref: RecordRef) -> str:
    # Deliberately identical to source_confirmatory_g01_gate.select_c0_refs.
    return ref_hash(ref, seed, "c0_record")


def scan_source(root: Path, logger: logging.Logger) -> dict[str, dict[str, list[RecordRef]]]:
    result: dict[str, dict[str, list[RecordRef]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted((root / "chunk_index").glob("x_chunk_*.json"))
    if not paths:
        raise FileNotFoundError(f"no source chunk indexes under {root}")
    for path_idx, path in enumerate(paths):
        chunk_idx = int(path.stem.rsplit("_", 1)[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("records", []):
            ref = parse_ref(raw, chunk_idx)
            if ref is not None:
                result[ref.source_key][ref.primary_dataset].append(ref)
        if path_idx == 0 or (path_idx + 1) % 500 == 0 or path_idx + 1 == len(paths):
            logger.info(
                "stage=index_scan files=%s/%s sources=%s records=%s",
                path_idx + 1,
                len(paths),
                len(result),
                sum(len(refs) for by_dataset in result.values() for refs in by_dataset.values()),
            )
    for source_key, by_dataset in result.items():
        for dataset, refs in by_dataset.items():
            refs.sort(key=lambda ref: record_order(260816, ref))
            if len({(ref.chunk_idx, ref.record_idx) for ref in refs}) != len(refs):
                raise RuntimeError(f"duplicate refs: {source_key}/{dataset}")
    return result


def selected_keys(
    ranked: dict[str, list[tuple[str, RecordRef]]],
    *,
    dataset_cap: int | None,
    record_cap: int,
) -> set[tuple[str, int, int]]:
    selected: set[tuple[str, int, int]] = set()
    for source_key, items in ranked.items():
        datasets: list[str] = []
        for dataset, _ref in items:
            if dataset not in datasets:
                datasets.append(dataset)
        if dataset_cap is not None:
            datasets = datasets[:dataset_cap]
        per_dataset: dict[str, int] = defaultdict(int)
        for dataset, ref in items:
            if dataset not in datasets or per_dataset[dataset] >= record_cap:
                continue
            selected.add((source_key, ref.chunk_idx, ref.record_idx))
            per_dataset[dataset] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only support audit for nested source conditioning-template estimators"
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=260816)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()
    if args.k_max < 4:
        raise ValueError("k-max must be at least 4")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "support_audit.log", mode="w")],
        force=True,
    )
    logger = logging.getLogger("source_condition_density_support")
    resolved = {
        "source_root": str(args.source_root.resolve()),
        "output_dir": str(output),
        "seed": args.seed,
        "k_max": args.k_max,
        "device": "CPU_ONLY_INDEX_AUDIT",
        "target_access": False,
        "dataset_order_contract": "stable_hex(seed, 'c0_dataset', source_key, dataset)",
        "record_order_contract": "ref_hash(ref, seed, 'c0_record')",
    }
    write_json(output / "resolved_config.json", resolved)
    logger.info("resolved_config=%s", json.dumps(resolved, sort_keys=True))

    by_source = scan_source(args.source_root, logger)
    ranked: dict[str, list[tuple[str, RecordRef]]] = {}
    for source_key, by_dataset in by_source.items():
        datasets = sorted(by_dataset, key=lambda dataset: dataset_order(args.seed, source_key, dataset))
        items: list[tuple[str, RecordRef]] = []
        for dataset in datasets:
            refs = sorted(by_dataset[dataset], key=lambda ref: record_order(args.seed, ref))
            items.extend((dataset, ref) for ref in refs)
        ranked[source_key] = items

    variants: list[tuple[str, int | None, int]] = [
        ("d2_k1", 2, 1),
        ("d2_k4", 2, 4),
        ("all_k1", None, 1),
        ("all_k4", None, 4),
    ]
    if args.k_max >= 8:
        variants.extend((("d2_k8", 2, 8), ("all_k8", None, 8)))
    variant_keys = {
        name: selected_keys(ranked, dataset_cap=dataset_cap, record_cap=record_cap)
        for name, dataset_cap, record_cap in variants
    }
    cache_keys = variant_keys[f"all_k{args.k_max}"]
    cache_rows: list[dict[str, Any]] = []
    pair_cache_counts: Counter[tuple[str, str]] = Counter()
    for source_key in sorted(ranked):
        datasets = sorted(by_source[source_key], key=lambda dataset: dataset_order(args.seed, source_key, dataset))
        for dataset_rank, dataset in enumerate(datasets):
            refs = sorted(by_source[source_key][dataset], key=lambda ref: record_order(args.seed, ref))
            for record_rank, ref in enumerate(refs[: args.k_max]):
                key = (source_key, ref.chunk_idx, ref.record_idx)
                if key not in cache_keys:
                    continue
                if ref.weight_shape[0] % 64:
                    raise RuntimeError(f"non-divisible source d_in: {ref}")
                row_groups = ref.weight_shape[0] // 64
                pair_cache_counts[(source_key, dataset)] += 1
                cache_rows.append(
                    {
                        "vector_index": len(cache_rows),
                        "source_key": source_key,
                        "model_name": ref.model_name,
                        "role": ref.role,
                        "depth": ref.depth,
                        "dataset": dataset,
                        "dataset_rank": dataset_rank,
                        "available_datasets_for_source": len(by_source[source_key]),
                        "record_rank": record_rank,
                        "available_records_for_pair": len(refs),
                        "chunk_idx": ref.chunk_idx,
                        "record_idx": ref.record_idx,
                        "weight_shape": f"{ref.weight_shape[0]}x{ref.weight_shape[1]}",
                        "row_groups": row_groups,
                        "distribution_encoder_batches_at_32_groups": math.ceil(row_groups / 32),
                        **{
                            f"used_{name}": key in keys
                            for name, keys in variant_keys.items()
                        },
                    }
                )
    write_csv(output / "condition_vector_cache_manifest.csv", cache_rows)

    variant_rows: list[dict[str, Any]] = []
    for name, dataset_cap, record_cap in variants:
        keys = variant_keys[name]
        rows = [row for row in cache_rows if row[f"used_{name}"]]
        pair_counts = Counter((row["source_key"], row["dataset"]) for row in rows)
        eligible_rows = [row for row in rows if int(row["available_datasets_for_source"]) >= 2]
        variant_rows.append(
            {
                "variant": name,
                "dataset_cap": "all" if dataset_cap is None else dataset_cap,
                "record_cap_per_dataset": record_cap,
                "selected_records_all_sources": len(keys),
                "selected_records_cell_eligible_sources": len(eligible_rows),
                "selected_source_dataset_pairs": len(pair_counts),
                "pairs_reaching_record_cap": sum(value == record_cap for value in pair_counts.values()),
                "pairs_below_record_cap": sum(value < record_cap for value in pair_counts.values()),
                "selected_sources": len({row["source_key"] for row in rows}),
                "cell_eligible_sources": len({row["source_key"] for row in eligible_rows}),
                "row_groups_all_sources": sum(int(row["row_groups"]) for row in rows),
                "distribution_encoder_batches_all_sources": sum(
                    int(row["distribution_encoder_batches_at_32_groups"]) for row in rows
                ),
                "raw_float32_vector_bytes_cvar_plus_cpatch": len(rows) * 512 * 4,
            }
        )
    write_csv(output / "variant_support_summary.csv", variant_rows)

    pair_counts_all = Counter(
        len(refs) for by_dataset in by_source.values() for refs in by_dataset.values()
    )
    datasets_per_source = Counter(len(by_dataset) for by_dataset in by_source.values())
    model_rows: list[dict[str, Any]] = []
    source_identity = {
        source_key: next(iter(next(iter(by_dataset.values()))))
        for source_key, by_dataset in by_source.items()
    }
    for model_name in sorted({ref.model_name for ref in source_identity.values()}):
        source_keys = [
            source_key for source_key, ref in source_identity.items() if ref.model_name == model_name
        ]
        model_rows.append(
            {
                "model_name": model_name,
                "sources": len(source_keys),
                "source_dataset_pairs": sum(len(by_source[source_key]) for source_key in source_keys),
                "available_records": sum(
                    len(refs)
                    for source_key in source_keys
                    for refs in by_source[source_key].values()
                ),
                "datasets_per_source_histogram": json.dumps(
                    dict(sorted(Counter(len(by_source[source_key]) for source_key in source_keys).items()))
                ),
                "records_per_pair_histogram": json.dumps(
                    dict(
                        sorted(
                            Counter(
                                len(refs)
                                for source_key in source_keys
                                for refs in by_source[source_key].values()
                            ).items()
                        )
                    )
                ),
            }
        )
    write_csv(output / "model_support_summary.csv", model_rows)

    summary = {
        "source_only": True,
        "target_access": False,
        "source_count": len(by_source),
        "source_dataset_pair_count": sum(len(value) for value in by_source.values()),
        "available_activation_record_count": sum(
            len(refs) for by_dataset in by_source.values() for refs in by_dataset.values()
        ),
        "datasets_per_source_histogram": dict(sorted(datasets_per_source.items())),
        "records_per_source_dataset_pair_histogram": dict(sorted(pair_counts_all.items())),
        "cache": {
            "k_max": args.k_max,
            "records": len(cache_rows),
            "raw_float32_vector_bytes_cvar_plus_cpatch": len(cache_rows) * 512 * 4,
            "raw_float32_vector_mib_cvar_plus_cpatch": len(cache_rows) * 512 * 4 / 2**20,
            "row_groups": sum(int(row["row_groups"]) for row in cache_rows),
            "distribution_encoder_batches_at_32_groups": sum(
                int(row["distribution_encoder_batches_at_32_groups"]) for row in cache_rows
            ),
        },
        "variants": variant_rows,
        "nested_checks": {
            "d2_k1_subset_d2_k4": variant_keys["d2_k1"] <= variant_keys["d2_k4"],
            "d2_k4_subset_all_k4": variant_keys["d2_k4"] <= variant_keys["all_k4"],
            "all_k1_subset_all_k4": variant_keys["all_k1"] <= variant_keys["all_k4"],
            **(
                {
                    "d2_k4_subset_d2_k8": variant_keys["d2_k4"] <= variant_keys["d2_k8"],
                    "all_k4_subset_all_k8": variant_keys["all_k4"] <= variant_keys["all_k8"],
                }
                if args.k_max >= 8
                else {}
            ),
        },
        "artifacts": [
            "condition_vector_cache_manifest.csv",
            "variant_support_summary.csv",
            "model_support_summary.csv",
            "resolved_config.json",
            "support_audit.log",
        ],
    }
    write_json(output / "support_inventory.json", summary)
    logger.info(
        "stage=output_writing cache_records=%s raw_vector_mib=%.3f artifacts=%s",
        len(cache_rows),
        summary["cache"]["raw_float32_vector_mib_cvar_plus_cpatch"],
        summary["artifacts"],
    )


if __name__ == "__main__":
    main()
