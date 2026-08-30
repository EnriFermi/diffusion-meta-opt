"""Frozen method×dataset×seed×protocol coverage grid validation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .contract import CandidateProtocol, DEFAULT_CONTRACT


GRID_KEYS = ("method", "dataset", "evaluation_seed", "protocol")
FINAL_EVALUATION_SEEDS = (0, 42, 777)
FINAL_METHOD_PROTOCOLS = (
    ("scratch", "controlled_single"),
    ("anchor_untouched", "controlled_single"),
    ("anchor", "controlled_single"),
    *( 
        (method, protocol)
        for method in (
            "ours_flow",
            "weightclip_flow",
            "ours_flow_oracle_anchor",
            "weightclip_flow_oracle_anchor",
        )
        for protocol in (
            "controlled_single",
            "controlled_validation_best_k",
            "native_test_top5_oracle",
        )
    ),
    *(
        (f"weightclip_commonzoo_fullwindow_{mode}", protocol)
        for mode in ("ridge", "memory", "nearest_code")
        for protocol in ("controlled_single", "controlled_validation_best_k")
    ),
    *(
        (f"weightclip_commonzoo_fullwindow_{mode}_native_oracle", "native_test_top5_oracle")
        for mode in ("ridge", "memory", "nearest_code")
    ),
)


def canonical_grid_cells(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = {
        (
            str(row["method"]),
            str(row["dataset"]),
            int(row["evaluation_seed"]),
            str(row["protocol"]),
        )
        for row in rows
    }
    return [dict(zip(GRID_KEYS, values, strict=True)) for values in sorted(cells)]


def canonical_final_grid_cells() -> list[dict[str, Any]]:
    return canonical_grid_cells(
        {
            "method": method,
            "dataset": dataset,
            "evaluation_seed": seed,
            "protocol": protocol,
        }
        for method, protocol in FINAL_METHOD_PROTOCOLS
        for dataset in DEFAULT_CONTRACT.ood_datasets
        for seed in FINAL_EVALUATION_SEEDS
    )


def validate_exact_grid(rows: Iterable[Mapping[str, Any]], expected: Iterable[Mapping[str, Any]]) -> None:
    actual_cells = canonical_grid_cells(rows)
    expected_cells = canonical_grid_cells(expected)
    if actual_cells != expected_cells:
        actual = {tuple(row[key] for key in GRID_KEYS) for row in actual_cells}
        wanted = {tuple(row[key] for key in GRID_KEYS) for row in expected_cells}
        raise ValueError(
            f"evaluation coverage grid mismatch: missing={sorted(wanted - actual)} extra={sorted(actual - wanted)}"
        )


def validate_candidate_cardinality(rows: Iterable[Mapping[str, Any]]) -> None:
    groups: dict[tuple[str, str, int, str], list[str]] = {}
    for row in rows:
        key = tuple(row[name] for name in GRID_KEYS)
        normalized = (str(key[0]), str(key[1]), int(key[2]), str(key[3]))
        groups.setdefault(normalized, []).append(str(row["candidate_id"]))
    for key, candidate_ids in groups.items():
        protocol = CandidateProtocol(key[3])
        expected = (
            1
            if protocol == CandidateProtocol.CONTROLLED_SINGLE
            else DEFAULT_CONTRACT.evaluation.controlled_candidate_count
            if protocol == CandidateProtocol.CONTROLLED_VALIDATION_BEST_K
            else DEFAULT_CONTRACT.evaluation.native_candidate_count
        )
        if len(candidate_ids) != expected or len(set(candidate_ids)) != expected:
            raise ValueError(
                f"candidate cardinality mismatch for {key}: rows={len(candidate_ids)} "
                f"unique={len(set(candidate_ids))} expected={expected}"
            )
