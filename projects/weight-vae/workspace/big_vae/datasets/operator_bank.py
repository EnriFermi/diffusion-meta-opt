"""Generic BigVAE data mode for content-addressed WeightCLIP operator banks.

Gauge augmentation is deliberately performed after reconstructing a complete
operator and its complete input context.  A permutation may cross 128-wide
tile boundaries, so independently permuting mmap tiles would be incorrect.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from big_vae.weightclip_benchmark.manifests import (
    sha256_file,
    validate_operator_bank_builder_source_seal,
)
from big_vae.weightclip_benchmark.parameter_adapters import (
    OperatorSpec,
    input_feature_permutation,
    output_feature_permutation,
)


def validate_array_bank_contents(root: str | Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Fail closed on the exact manifest-declared array-bank payload inventory."""

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != str(expected_manifest_sha256):
        raise RuntimeError(f"operator array-bank manifest is missing or corrupt: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_payload = dict(manifest)
    declared_content = str(content_payload.pop("content_sha256", ""))
    actual_content = hashlib.sha256(
        json.dumps(content_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if declared_content != actual_content or not root.name.endswith(declared_content[:16]):
        raise RuntimeError(f"operator array-bank content address mismatch: {root}")
    declared_paths: set[Path] = set()
    declared_shard_dirs: set[Path] = set()
    declared_shard_indices: set[int] = set()
    for shard in manifest.get("shards", []):
        shard_index = int(shard.get("index", -1))
        if shard_index < 0 or shard_index in declared_shard_indices:
            raise RuntimeError(f"operator array-bank shard index is invalid or duplicated: {shard_index}")
        declared_shard_indices.add(shard_index)
        expected_shard_dir = (root / f"shard-{shard_index:06d}").resolve()
        declared_shard_dirs.add(expected_shard_dir)
        files = shard.get("files", {})
        if not isinstance(files, dict) or not files:
            raise RuntimeError(f"operator array-bank shard has no file inventory: {root}")
        for file_row in files.values():
            path = (root / str(file_row["path"])).resolve()
            if (
                not path.is_relative_to(root)
                or path.parent != expected_shard_dir
                or path in declared_paths
                or not path.is_file()
            ):
                raise RuntimeError(f"operator array-bank shard path is invalid or duplicated: {path}")
            declared_paths.add(path)
            if path.stat().st_size != int(file_row["bytes"]):
                raise RuntimeError(f"operator array-bank shard size mismatch: {path}")
            if sha256_file(path) != str(file_row["sha256"]):
                raise RuntimeError(f"operator array-bank shard hash mismatch: {path}")
    if not declared_paths:
        raise RuntimeError(f"operator array-bank manifest declares no payload files: {manifest_path}")
    expected_indices = set(range(len(declared_shard_indices)))
    if declared_shard_indices != expected_indices:
        raise RuntimeError(
            f"operator array-bank shard indices must be contiguous: "
            f"declared={sorted(declared_shard_indices)} expected={sorted(expected_indices)}"
        )
    actual_shard_entries = {path.resolve() for path in root.glob("shard-*")}
    if actual_shard_entries != declared_shard_dirs or any(not path.is_dir() for path in actual_shard_entries):
        raise RuntimeError(
            f"operator array-bank shard directory inventory mismatch: root={root} "
            f"declared={sorted(map(str, declared_shard_dirs))} actual={sorted(map(str, actual_shard_entries))}"
        )
    actual_payload_paths: set[Path] = set()
    for shard_dir in actual_shard_entries:
        actual_payload_paths.update(path.resolve() for path in shard_dir.iterdir())
    if actual_payload_paths != declared_paths or any(not path.is_file() for path in actual_payload_paths):
        raise RuntimeError(
            f"operator array-bank payload inventory mismatch: root={root} "
            f"declared={sorted(map(str, declared_paths))} actual={sorted(map(str, actual_payload_paths))}"
        )
    return manifest


@dataclass(slots=True)
class OperatorBankSample:
    x: torch.Tensor
    weight: torch.Tensor
    meta: dict[str, Any]
    model_name: str
    layer_name: str


@dataclass(frozen=True, slots=True)
class OperatorBundleRequest:
    """One full operator/gauge materialization requested from a loader worker."""

    cycle: int
    key: tuple[str, str]


@dataclass(slots=True)
class OperatorBankBundle:
    """Full permuted operator transferred once; tiling happens in the main process."""

    request: OperatorBundleRequest
    matrix: torch.Tensor
    context: torch.Tensor
    sample_mask: torch.Tensor
    spec: OperatorSpec
    group_meta: dict[str, Any]
    gauge_id: str
    gauge_view_index: int

    @property
    def tile_count(self) -> int:
        rows, cols = self.spec.matrix_shape
        return ((rows + 127) // 128) * ((cols + 127) // 128)

    @property
    def tensor_bytes(self) -> int:
        tensors = (self.matrix, self.context, self.sample_mask)
        return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


@dataclass(frozen=True, slots=True)
class _RecordLocation:
    shard: int
    offset: int
    metadata: dict[str, Any]


def _operator_spec(payload: Mapping[str, Any]) -> OperatorSpec:
    values = dict(payload)
    for key in ("matrix_shape", "native_shape", "kernel", "stride", "padding", "dilation"):
        values[key] = tuple(values[key])
    return OperatorSpec(**values)


class _MMapBank:
    def __init__(self, root: Path, *, max_open_shards: int, expected_manifest_sha256: str) -> None:
        self.root = root.resolve()
        manifest = validate_array_bank_contents(self.root, expected_manifest_sha256)
        self.max_open_shards = max(1, int(max_open_shards))
        self.shards = [
            self.root / f"shard-{int(shard['index']):06d}"
            for shard in sorted(manifest["shards"], key=lambda row: int(row["index"]))
        ]
        if not self.shards:
            raise ValueError(f"bank has no shards: {root}")
        self.records: list[_RecordLocation] = []
        for shard_index, shard in enumerate(self.shards):
            for offset, line in enumerate((shard / "records.jsonl").read_text().splitlines()):
                self.records.append(_RecordLocation(shard_index, offset, json.loads(line)))
        self._arrays: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()

    def array(self, location: _RecordLocation, field: str) -> np.ndarray:
        key = (location.shard, field)
        if key not in self._arrays:
            self._arrays[key] = np.load(self.shards[location.shard] / f"{field}.npy", mmap_mode="r", allow_pickle=False)
            while len(self._arrays) > self.max_open_shards:
                self._arrays.popitem(last=False)
        else:
            self._arrays.move_to_end(key)
        return self._arrays[key][location.offset]


class OperatorBankTrainingDataset(Dataset[OperatorBankSample]):
    """Step-addressable full-operator stream with graph-gauge augmentation."""

    def __init__(
        self,
        pair_manifest: str | Path,
        *,
        seed: int,
        repeat: bool = True,
        permutation_views: bool = True,
        canonical_probability: float = 1.0 / 6.0,
        hot_shards: int = 8,
        expected_pair_manifest_sha256: str = "",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.pair_manifest_path = Path(pair_manifest).resolve()
        if expected_pair_manifest_sha256:
            actual_pair_sha = sha256_file(self.pair_manifest_path)
            if actual_pair_sha != str(expected_pair_manifest_sha256):
                raise ValueError(
                    f"operator-bank pair manifest checksum mismatch: {self.pair_manifest_path}"
                )
        pair = json.loads(self.pair_manifest_path.read_text())
        source_seal = pair.get("contract", {}).get("builder_source_implementation")
        if not isinstance(source_seal, dict):
            raise RuntimeError("operator-bank pair manifest lacks a builder source implementation seal")
        validate_operator_bank_builder_source_seal(source_seal)
        self.context_bank = _MMapBank(
            Path(pair["context_bank"]),
            max_open_shards=hot_shards,
            expected_manifest_sha256=str(pair["context_bank_manifest_sha256"]),
        )
        self.weight_bank = _MMapBank(
            Path(pair["weight_tile_bank"]),
            max_open_shards=hot_shards,
            expected_manifest_sha256=str(pair["weight_tile_bank_manifest_sha256"]),
        )
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.permutation_views = bool(permutation_views)
        self.canonical_probability = float(canonical_probability)
        if not 0.0 <= self.canonical_probability <= 1.0:
            raise ValueError("canonical_probability must lie in [0, 1]")
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("invalid rank/world_size")

        self.context_by_id = {
            location.metadata["context_id"]: location for location in self.context_bank.records
        }
        self.operator_groups: dict[tuple[str, str], list[_RecordLocation]] = {}
        for location in self.weight_bank.records:
            metadata = location.metadata
            key = (metadata["checkpoint_sha256"], metadata["layer_key"])
            self.operator_groups.setdefault(key, []).append(location)
        self.views_by_checkpoint = self._load_views(pair)
        if self.permutation_views:
            missing = sorted({checkpoint for checkpoint, _ in self.operator_groups} - set(self.views_by_checkpoint))
            if missing:
                raise ValueError(f"missing graph-gauge views for {len(missing)} checkpoints; first={missing[0]}")
            wrong = {key: len(value) for key, value in self.views_by_checkpoint.items() if len(value) != 5}
            if wrong:
                raise ValueError(f"the approved operator-bank mode requires exactly five views/checkpoint: {wrong}")
        self._keys = sorted(self.operator_groups)[self.rank :: self.world_size]
        self._records_per_cycle = sum(len(self.operator_groups[key]) for key in self._keys)
        if self._records_per_cycle <= 0:
            raise ValueError(f"rank {self.rank}/{self.world_size} has no operator-bank records")
        self._cycle_cache: OrderedDict[
            int,
            tuple[list[tuple[tuple[str, str], int]], dict[str, dict[str, Any] | None]],
        ] = OrderedDict()
        self._group_cache: OrderedDict[tuple[int, tuple[str, str]], list[OperatorBankSample]] = OrderedDict()
        self._locality_schedule_cache: OrderedDict[
            tuple[int, int], tuple[tuple[tuple[str, str], int], ...]
        ] = OrderedDict()

    @staticmethod
    def _load_views(pair: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        files = pair.get("permutation_files", {})
        jsonl_paths = [Path(path) for path in files if str(path).endswith(".jsonl")]
        if not jsonl_paths:
            return {}
        if len(jsonl_paths) != 1:
            raise ValueError(f"expected one permutation JSONL, found {len(jsonl_paths)}")
        expected = files[str(jsonl_paths[0])]
        actual = sha256_file(jsonl_paths[0])
        if actual != expected:
            raise ValueError(f"permutation-view checksum mismatch: {jsonl_paths[0]}")
        rows = [json.loads(line) for line in jsonl_paths[0].read_text().splitlines() if line]
        return {row["checkpoint_sha256"]: row["views"] for row in rows}

    def __len__(self) -> int:
        return self._records_per_cycle

    def cache_size(self) -> int:
        """Compatibility metric consumed by the generic worker telemetry."""

        return len(self.context_bank._arrays) + len(self.weight_bank._arrays)

    def _full_operator(self, locations: list[_RecordLocation]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, OperatorSpec, dict[str, Any]]:
        first = locations[0].metadata
        spec = _operator_spec(first["operator"])
        matrix = torch.zeros(spec.matrix_shape, dtype=torch.float32)
        context = torch.zeros((512, spec.matrix_shape[0]), dtype=torch.float32)
        sample_mask: torch.Tensor | None = None
        seen_contexts: set[str] = set()
        for location in locations:
            tile_meta = location.metadata["tile"]
            rs, cs = int(tile_meta["row_start"]), int(tile_meta["col_start"])
            vr, vc = int(tile_meta["valid_rows"]), int(tile_meta["valid_cols"])
            tile = torch.from_numpy(np.array(self.weight_bank.array(location, "weight"), copy=True))
            matrix[rs : rs + vr, cs : cs + vc] = tile[:vr, :vc]
            context_id = location.metadata["context_id"]
            if context_id in seen_contexts:
                continue
            seen_contexts.add(context_id)
            context_location = self.context_by_id[context_id]
            context_meta = context_location.metadata
            start = int(context_meta["input_row_start"])
            valid = int(context_meta["valid_features"])
            raw = torch.from_numpy(np.array(self.context_bank.array(context_location, "raw_rows"), copy=True))
            current_mask = torch.from_numpy(np.array(self.context_bank.array(context_location, "sample_mask"), copy=True)).bool()
            if sample_mask is None:
                sample_mask = current_mask
            elif not torch.equal(sample_mask, current_mask):
                raise ValueError(f"context sample masks disagree inside operator {spec.key}")
            context[:, start : start + valid] = raw[:, :valid]
        if sample_mask is None:
            raise ValueError(f"operator {spec.key} has no context records")
        return matrix, context, sample_mask, spec, first

    def _materialize_bundle(
        self,
        request: OperatorBundleRequest,
        view: dict[str, Any] | None,
    ) -> OperatorBankBundle:
        key = request.key
        matrix, context, sample_mask, spec, group_meta = self._full_operator(self.operator_groups[key])
        gauge_id = "canonical"
        view_index = -1
        if view is not None:
            gauge = {name: torch.tensor(indices, dtype=torch.long) for name, indices in view["groups"].items()}
            input_perm = input_feature_permutation(spec, gauge)
            output_perm = output_feature_permutation(spec, gauge)
            # Both permutations are applied before re-tiling.  They can cross
            # mmap tile boundaries and therefore cannot be done tile-locally.
            matrix = matrix.index_select(0, input_perm).index_select(1, output_perm)
            context = context.index_select(1, input_perm)
            gauge_id = str(view["gauge_id"])
            view_index = int(view["view_index"])
        return OperatorBankBundle(
            request=request,
            matrix=matrix,
            context=context,
            sample_mask=sample_mask,
            spec=spec,
            group_meta=group_meta,
            gauge_id=gauge_id,
            gauge_view_index=view_index,
        )

    @staticmethod
    def _sample_from_bundle(bundle: OperatorBankBundle, local_index: int) -> OperatorBankSample:
        rows, cols = bundle.spec.matrix_shape
        tile_cols = (cols + 127) // 128
        tile_row, tile_col = divmod(int(local_index), tile_cols)
        row_start, col_start = tile_row * 128, tile_col * 128
        valid_rows = min(128, rows - row_start)
        valid_cols = min(128, cols - col_start)
        if local_index < 0 or row_start >= rows or col_start >= cols:
            raise IndexError(f"invalid local tile index {local_index} for {bundle.spec.key}")
        tile = torch.zeros((128, 128), dtype=torch.float32)
        tile[:valid_rows, :valid_cols] = bundle.matrix[
            row_start : row_start + valid_rows,
            col_start : col_start + valid_cols,
        ]
        x = torch.zeros((512, 128), dtype=torch.float32)
        x[:, :valid_rows] = bundle.context[:, row_start : row_start + valid_rows]
        d_in_mask = torch.zeros(128, dtype=torch.bool)
        d_in_mask[:valid_rows] = True
        d_out_mask = torch.zeros(128, dtype=torch.bool)
        d_out_mask[:valid_cols] = True
        metadata = {
            "x_mask": bundle.sample_mask.clone(),
            "d_in_mask": d_in_mask,
            "d_out_mask": d_out_mask,
            "dataset": bundle.group_meta.get("dataset", "unspecified_dataset"),
            "lineage_id": bundle.group_meta["lineage_id"],
            "checkpoint_sha256": bundle.group_meta["checkpoint_sha256"],
            "layer_key": bundle.spec.key,
            "operator": bundle.spec.to_dict(),
            "tile_row": tile_row,
            "tile_col": tile_col,
            "tile_row_start": row_start,
            "tile_col_start": col_start,
            "gauge_id": bundle.gauge_id,
            "gauge_view_index": bundle.gauge_view_index,
            "augmentation_is_independent_lineage": False,
            "full_operator_grouped_before_permutation": True,
            "bundle_cycle": bundle.request.cycle,
        }
        return OperatorBankSample(
            x=x,
            weight=tile,
            meta=metadata,
            model_name=str(bundle.group_meta["lineage_id"]),
            layer_name=bundle.spec.key,
        )

    def _materialize_group(self, key: tuple[str, str], view: dict[str, Any] | None) -> list[OperatorBankSample]:
        request = OperatorBundleRequest(cycle=0, key=key)
        bundle = self._materialize_bundle(request, view)
        return [self._sample_from_bundle(bundle, index) for index in range(bundle.tile_count)]

    def _cycle_plan(
        self,
        cycle: int,
    ) -> tuple[list[tuple[tuple[str, str], int]], dict[str, dict[str, Any] | None]]:
        if cycle in self._cycle_cache:
            self._cycle_cache.move_to_end(cycle)
            return self._cycle_cache[cycle]
        rng = random.Random(self.seed + 1_000_003 * cycle + self.rank)
        cycle_keys = list(self._keys)
        rng.shuffle(cycle_keys)
        chosen_views: dict[str, dict[str, Any] | None] = {}
        plan: list[tuple[tuple[str, str], int]] = []
        for checkpoint, _ in cycle_keys:
            if checkpoint not in chosen_views:
                # WeightCLIP returns canonical + five random permutations.
                use_canonical = (not self.permutation_views) or (
                    rng.random() < self.canonical_probability
                )
                chosen_views[checkpoint] = (
                    None if use_canonical else rng.choice(self.views_by_checkpoint[checkpoint])
                )
        # Hierarchical stratum balance prevents large tiled layers from
        # dominating merely because they contain more 128x128 tiles.  The
        # joint key preserves equal dataset/op/depth/role exposure; records
        # inside each stratum are sampled cyclically with replacement.  The
        # total logical cycle length remains unchanged for exact resume math.
        strata: dict[tuple[str, str, int, str], list[tuple[tuple[str, str], int]]] = {}
        for key in cycle_keys:
            locations = self.operator_groups[key]
            operator = locations[0].metadata["operator"]
            stratum = (
                str(locations[0].metadata.get("dataset", "unspecified_dataset")),
                str(operator["operation"]),
                int(operator["depth_index"]),
                str(operator["role"]),
            )
            bucket = strata.setdefault(stratum, [])
            bucket.extend((key, local_index) for local_index in range(len(locations)))
        stratum_keys = sorted(strata)
        rng.shuffle(stratum_keys)
        for bucket in strata.values():
            rng.shuffle(bucket)
        cursors = {stratum: 0 for stratum in stratum_keys}
        while len(plan) < self._records_per_cycle:
            for stratum in stratum_keys:
                bucket = strata[stratum]
                plan.append(bucket[cursors[stratum] % len(bucket)])
                cursors[stratum] += 1
                if len(plan) == self._records_per_cycle:
                    break
        result = (plan, chosen_views)
        self._cycle_cache[cycle] = result
        while len(self._cycle_cache) > 2:
            self._cycle_cache.popitem(last=False)
        return result

    def _stratum_for_key(self, key: tuple[str, str]) -> tuple[str, str, int, str]:
        location = self.operator_groups[key][0]
        operator = location.metadata["operator"]
        return (
            str(location.metadata.get("dataset", "unspecified_dataset")),
            str(operator["operation"]),
            int(operator["depth_index"]),
            str(operator["role"]),
        )

    def locality_plan(self, cycle: int) -> tuple[tuple[tuple[str, str], int], ...]:
        """Preserve the exact global stratum sequence; group only within strata."""

        cache_key = (int(cycle), 0)
        cached = self._locality_schedule_cache.get(cache_key)
        if cached is not None:
            self._locality_schedule_cache.move_to_end(cache_key)
            return cached
        target_plan, _views = self._cycle_plan(cycle)
        by_stratum: dict[
            tuple[str, str, int, str],
            dict[tuple[str, str], list[int]],
        ] = {}
        key_order: dict[tuple[str, str, int, str], list[tuple[str, str]]] = {}
        stratum_sequence: list[tuple[str, str, int, str]] = []
        for key, local_index in target_plan:
            stratum = self._stratum_for_key(key)
            stratum_sequence.append(stratum)
            buckets = by_stratum.setdefault(stratum, {})
            if key not in buckets:
                buckets[key] = []
                key_order.setdefault(stratum, []).append(key)
            buckets[key].append(local_index)
        grouped_iterators: dict[
            tuple[str, str, int, str],
            Iterator[tuple[tuple[str, str], int]],
        ] = {}
        for stratum, ordered_keys in key_order.items():
            grouped_entries = tuple(
                (key, local_index)
                for key in ordered_keys
                for local_index in by_stratum[stratum][key]
            )
            grouped_iterators[stratum] = iter(grouped_entries)
        entries = [next(grouped_iterators[stratum]) for stratum in stratum_sequence]
        if len(entries) != self._records_per_cycle:
            raise RuntimeError("operator-bank locality schedule changed the target cycle size")
        result = tuple(entries)
        self._locality_schedule_cache[cache_key] = result
        while len(self._locality_schedule_cache) > 2:
            self._locality_schedule_cache.popitem(last=False)
        return result

    def _plan_order_metrics(
        self,
        plan: Sequence[tuple[tuple[str, str], int]],
        *,
        batch_size: int = 32,
    ) -> dict[str, float | int]:
        lineages = [str(self.operator_groups[key][0].metadata["lineage_id"]) for key, _ in plan]
        checkpoints = [key[0] for key, _ in plan]

        def batch_counts(values: Sequence[str]) -> list[int]:
            return [len(set(values[start : start + batch_size])) for start in range(0, len(values), batch_size)]

        def p5(values: Sequence[int]) -> float:
            ordered = sorted(values)
            return float(ordered[int(0.05 * max(0, len(ordered) - 1))]) if ordered else 0.0

        max_run = 0
        run = 0
        previous: str | None = None
        for lineage in lineages:
            run = run + 1 if lineage == previous else 1
            max_run = max(max_run, run)
            previous = lineage
        lineage_counts = batch_counts(lineages)
        checkpoint_counts = batch_counts(checkpoints)
        adjacent_same = sum(left == right for left, right in zip(lineages, lineages[1:]))
        return {
            "batch_size": batch_size,
            "batches": len(lineage_counts),
            "unique_lineages_p5": p5(lineage_counts),
            "unique_lineages_mean": float(sum(lineage_counts) / max(1, len(lineage_counts))),
            "unique_checkpoints_p5": p5(checkpoint_counts),
            "unique_checkpoints_mean": float(sum(checkpoint_counts) / max(1, len(checkpoint_counts))),
            "max_same_lineage_run": max_run,
            "adjacent_same_lineage_fraction": float(adjacent_same / max(1, len(lineages) - 1)),
        }

    def locality_audit(self, cycle: int = 0) -> dict[str, Any]:
        target, _views = self._cycle_plan(cycle)
        locality = self.locality_plan(cycle)
        target_strata = [self._stratum_for_key(key) for key, _ in target]
        locality_strata = [self._stratum_for_key(key) for key, _ in locality]
        if Counter(target) != Counter(locality):
            raise RuntimeError("operator locality plan changed the exact target entry multiset")
        if target_strata != locality_strata:
            raise RuntimeError("operator locality plan changed the global dataset/op/depth/role sequence")
        target_lineages = [str(self.operator_groups[key][0].metadata["lineage_id"]) for key, _ in target]
        locality_lineages = [str(self.operator_groups[key][0].metadata["lineage_id"]) for key, _ in locality]
        return {
            "schema": "operator_bank_locality_audit_v1",
            "planning_algorithm": "global_stratum_sequence_exact_within_stratum_grouped_v1",
            "cycle": int(cycle),
            "records": len(target),
            "strata": len(set(target_strata)),
            "unique_groups": len({key for key, _ in target}),
            "exact_entry_multiset": True,
            "exact_global_stratum_sequence": True,
            "lineage_checkpoint_temporal_order_is_not_preserved_by_contract": True,
            "observed_lineage_temporal_order_equal": target_lineages == locality_lineages,
            "old_order": self._plan_order_metrics(target),
            "locality_order": self._plan_order_metrics(locality),
        }

    def __getitem__(self, logical_index: int) -> OperatorBankSample:
        logical_index = int(logical_index)
        if logical_index < 0:
            raise IndexError(logical_index)
        if not self.repeat and logical_index >= self._records_per_cycle:
            raise IndexError(logical_index)
        cycle, offset = divmod(logical_index, self._records_per_cycle)
        plan, views = self._cycle_plan(cycle)
        key, local_index = plan[offset]
        cache_key = (cycle, key)
        if cache_key not in self._group_cache:
            self._group_cache[cache_key] = self._materialize_group(key, views[key[0]])
            while len(self._group_cache) > 2:
                self._group_cache.popitem(last=False)
        else:
            self._group_cache.move_to_end(cache_key)
        return self._group_cache[cache_key][local_index]


class OperatorBankBundleDataset(Dataset[OperatorBankBundle]):
    """Worker-side dataset: each item reconstructs one full operator exactly once."""

    def __init__(
        self,
        source: OperatorBankTrainingDataset,
        *,
        max_active_strata: int,
        max_active_bundle_bytes: int,
    ) -> None:
        super().__init__()
        self.source = source
        self.max_active_strata = int(max_active_strata)
        self.max_active_bundle_bytes = int(max_active_bundle_bytes)
        if self.max_active_strata < 1:
            raise ValueError("max_active_strata must be positive")
        if self.max_active_bundle_bytes < 1:
            raise ValueError("max_active_bundle_bytes must be positive")
        self.source.locality_plan(0)
        self.locality_audit = self.source.locality_audit(0)
        strata = {self.source._stratum_for_key(key) for key in self.source.operator_groups}
        if len(strata) > self.max_active_strata:
            raise RuntimeError(
                f"operator bank has {len(strata)} strata, exceeding max_active_strata={self.max_active_strata}"
            )
        if self.active_tensor_bytes_bound > self.max_active_bundle_bytes:
            raise RuntimeError(
                "operator bank active-stratum tensor bound exceeds max_active_bundle_bytes: "
                f"bound={self.active_tensor_bytes_bound} max={self.max_active_bundle_bytes}"
            )

    def __len__(self) -> int:
        return len(self.source.operator_groups)

    @property
    def records_per_cycle(self) -> int:
        return len(self.source)

    def cache_size(self) -> int:
        return self.source.cache_size()

    @property
    def max_bundle_tensor_bytes(self) -> int:
        maximum = 0
        for locations in self.source.operator_groups.values():
            spec = _operator_spec(locations[0].metadata["operator"])
            rows, cols = spec.matrix_shape
            tensor_bytes = 4 * (rows * cols + 512 * rows) + 512
            maximum = max(maximum, tensor_bytes)
        return maximum

    @property
    def active_tensor_bytes_bound(self) -> int:
        per_stratum: dict[tuple[str, str, int, str], int] = {}
        for key, locations in self.source.operator_groups.items():
            spec = _operator_spec(locations[0].metadata["operator"])
            rows, cols = spec.matrix_shape
            tensor_bytes = 4 * (rows * cols + 512 * rows) + 512
            stratum = self.source._stratum_for_key(key)
            per_stratum[stratum] = max(per_stratum.get(stratum, 0), tensor_bytes)
        return sum(per_stratum.values())

    def __getitem__(self, request: OperatorBundleRequest) -> OperatorBankBundle:
        if not isinstance(request, OperatorBundleRequest):
            raise TypeError("bundle dataset indices must be OperatorBundleRequest objects")
        _plan, views = self.source._cycle_plan(request.cycle)
        if request.key not in self.source.operator_groups:
            raise KeyError(request.key)
        return self.source._materialize_bundle(request, views[request.key[0]])


class CommittedBundleSampler(Sampler[OperatorBundleRequest]):
    """Request complete locality windows starting at the committed tile cursor."""

    def __init__(self, dataset: OperatorBankBundleDataset, *, start_index: int = 0) -> None:
        self.dataset = dataset
        self.start_index = int(start_index)

    def set_start_index(self, value: int) -> None:
        if value < 0:
            raise ValueError("start_index must be non-negative")
        self.start_index = int(value)

    def __iter__(self) -> Iterator[OperatorBundleRequest]:
        records = self.dataset.records_per_cycle
        cycle, offset = divmod(self.start_index, records)
        while self.dataset.source.repeat or cycle == 0:
            plan = self.dataset.source.locality_plan(cycle)
            seen: set[tuple[str, str]] = set()
            for key, _local_index in plan[offset:]:
                if key not in seen:
                    seen.add(key)
                    yield OperatorBundleRequest(cycle=cycle, key=key)
            cycle += 1
            offset = 0

    def __len__(self) -> int:
        if self.dataset.source.repeat:
            return 2**63 - 1
        cycle, offset = divmod(self.start_index, self.dataset.records_per_cycle)
        if cycle > 0:
            return 0
        plan = self.dataset.source.locality_plan(0)
        return len({key for key, _local_index in plan[offset:]})


class BalancedOperatorBankMixer(Iterator[OperatorBankSample]):
    """Main-process tile mixer over bounded full-operator locality windows."""

    def __init__(
        self,
        dataset: OperatorBankBundleDataset,
        bundle_iter: Iterator[OperatorBankBundle],
        *,
        start_index: int,
    ) -> None:
        self.dataset = dataset
        self.bundle_iter = bundle_iter
        self.start_logical_index = int(start_index)
        self.logical_index = int(start_index)
        self._cycle = -1
        self._plan: Sequence[tuple[tuple[str, str], int]] = ()
        self._plan_offset = 0
        self._remaining: Counter[tuple[str, str]] = Counter()
        self._active: dict[tuple[str, str], OperatorBankBundle] = {}
        self.materialized_bundles = 0
        self.emitted_tiles = 0
        self.max_active_bundles = 0
        self.max_active_tensor_bytes = 0

    def __iter__(self) -> BalancedOperatorBankMixer:
        return self

    def _prepare_cycle(self) -> None:
        records = self.dataset.records_per_cycle
        cycle, offset = divmod(self.logical_index, records)
        self._cycle = cycle
        self._plan = self.dataset.source.locality_plan(cycle)
        self._plan_offset = offset
        self._remaining = Counter(key for key, _local_index in self._plan[offset:])
        self._active.clear()

    def _load_bundle(self, key: tuple[str, str]) -> OperatorBankBundle:
        bundle = next(self.bundle_iter)
        expected_request = OperatorBundleRequest(self._cycle, key)
        if bundle.request != expected_request:
            raise RuntimeError(
                f"operator bundle stream order mismatch: expected={expected_request} actual={bundle.request}"
            )
        _target, views = self.dataset.source._cycle_plan(self._cycle)
        expected_view = views[key[0]]
        expected_gauge = "canonical" if expected_view is None else str(expected_view["gauge_id"])
        if bundle.gauge_id != expected_gauge:
            raise RuntimeError("operator bundle gauge disagrees with the cycle checkpoint view")
        self._active[key] = bundle
        self.materialized_bundles += 1
        self.max_active_bundles = max(self.max_active_bundles, len(self._active))
        self.max_active_tensor_bytes = max(
            self.max_active_tensor_bytes,
            sum(active_bundle.tensor_bytes for active_bundle in self._active.values()),
        )
        if len(self._active) > self.dataset.max_active_strata:
            raise RuntimeError("operator locality mixer exceeded its bounded active-set contract")
        if self.max_active_tensor_bytes > self.dataset.max_active_bundle_bytes:
            raise RuntimeError("operator locality mixer exceeded its active tensor-byte contract")
        return bundle

    def __next__(self) -> OperatorBankSample:
        records = self.dataset.records_per_cycle
        cycle, _offset = divmod(self.logical_index, records)
        if cycle != self._cycle:
            self._prepare_cycle()
        key, local_index = self._plan[self._plan_offset]
        bundle = self._active.get(key)
        if bundle is None:
            bundle = self._load_bundle(key)
        sample = self.dataset.source._sample_from_bundle(bundle, local_index)
        # This is the committed-stream identity of the produced tile. It must
        # travel with the prepared batch because background prefetch can move
        # ``self.logical_index`` well beyond what the optimizer consumed.
        sample.meta["logical_index"] = int(self.logical_index)
        self._remaining[key] -= 1
        if self._remaining[key] == 0:
            del self._remaining[key]
            del self._active[key]
        self._plan_offset += 1
        self.logical_index += 1
        self.emitted_tiles += 1
        return sample

    def telemetry(self) -> dict[str, int | str | bool]:
        return {
            "schema": "operator_bank_mixer_telemetry_v1",
            "planning_algorithm": "global_stratum_sequence_exact_within_stratum_grouped_v1",
            "global_stratum_sequence_exact": True,
            "lineage_checkpoint_temporal_order_preserved": False,
            "start_logical_index": self.start_logical_index,
            "logical_index": self.logical_index,
            "records_per_cycle": self.dataset.records_per_cycle,
            "cycle": self.logical_index // self.dataset.records_per_cycle,
            "materialized_bundles": self.materialized_bundles,
            "emitted_tiles": self.emitted_tiles,
            "max_active_bundles": self.max_active_bundles,
            "max_active_tensor_bytes": self.max_active_tensor_bytes,
            "active_tensor_bytes_bound": self.dataset.active_tensor_bytes_bound,
            "max_bundle_tensor_bytes": self.dataset.max_bundle_tensor_bytes,
        }


class CommittedIndexSampler(Sampler[int]):
    """Regenerate the logical stream from the last committed optimizer step."""

    def __init__(self, dataset: OperatorBankTrainingDataset, *, start_index: int = 0) -> None:
        self.dataset = dataset
        self.start_index = int(start_index)

    def set_start_index(self, value: int) -> None:
        if value < 0:
            raise ValueError("start_index must be non-negative")
        self.start_index = int(value)

    def __iter__(self) -> Iterator[int]:
        index = self.start_index
        limit = None if self.dataset.repeat else len(self.dataset)
        while limit is None or index < limit:
            yield index
            index += 1

    def __len__(self) -> int:
        if self.dataset.repeat:
            return 2**63 - 1
        return max(0, len(self.dataset) - self.start_index)


@contextmanager
def operator_bank_data_pipeline(
    pair_manifest: str | Path,
    *,
    seed: int,
    repeat: bool,
    permutation_views: bool,
    canonical_probability: float,
    hot_shards: int,
    expected_pair_manifest_sha256: str,
    rank: int,
    world_size: int,
    max_active_strata: int = 256,
    max_active_bundle_bytes: int = 1536 * 1024 * 1024,
    logger: logging.Logger | None = None,
) -> Iterator[tuple[OperatorBankBundleDataset, CommittedBundleSampler]]:
    source = OperatorBankTrainingDataset(
        pair_manifest,
        seed=seed,
        repeat=repeat,
        permutation_views=permutation_views,
        canonical_probability=canonical_probability,
        hot_shards=hot_shards,
        expected_pair_manifest_sha256=expected_pair_manifest_sha256,
        rank=rank,
        world_size=world_size,
    )
    if logger is not None:
        logger.info(
            "Operator-bank opened: pair_manifest=%s canonical_tiles=%s operator_groups=%s "
            "five_graph_gauge_views=%s canonical_probability=%.6f hot_shards=%s rank=%s/%s",
            Path(pair_manifest).resolve(),
            len(source),
            len(source.operator_groups),
            permutation_views,
            canonical_probability,
            hot_shards,
            rank,
            world_size,
        )
    dataset = OperatorBankBundleDataset(
        source,
        max_active_strata=max_active_strata,
        max_active_bundle_bytes=max_active_bundle_bytes,
    )
    if logger is not None:
        logger.info(
            "Operator-bank locality mixer enabled: max_active_strata=%s groups_per_cycle=%s "
            "tile_records_per_cycle=%s planning=global_stratum_sequence_exact_within_stratum_grouped "
            "worker_materialization=full_operator main_process_tiling=true",
            max_active_strata,
            len(dataset),
            dataset.records_per_cycle,
        )
        logger.info("Operator-bank locality audit: %s", json.dumps(dataset.locality_audit, sort_keys=True))
    yield dataset, CommittedBundleSampler(dataset)
