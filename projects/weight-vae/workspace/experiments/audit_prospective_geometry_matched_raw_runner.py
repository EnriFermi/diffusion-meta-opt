#!/usr/bin/env python3
"""Independent, CPU-only audit of the prospective raw Weight-AE runner output.

This program deliberately does not import either the formal runner or the CPU
analyzer.  It verifies the immutable binding and raw sufficient-statistic
artifacts, reconstructs the tiling and permutation contracts, and computes
point aggregates directly from ``(T, P, D)``.  It never loads a Weight-AE or a
replication-panel tensor and must only be run after the formal runner has
written its terminal artifact manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch


PANEL_IDS = ("beans", "trocr_sroie")
SPLITS = ("A", "B")
TILING_SEEDS = (26081801, 26081802)
ARMS = ("correct", "permuted_within_row", "zero_code")
ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)
ROLE_SHAPES = {
    "attn_query": (768, 768),
    "attn_key": (768, 768),
    "attn_value": (768, 768),
    "attn_output": (768, 768),
    "ffn_up": (768, 3072),
    "ffn_down": (3072, 768),
}
ROLE_GAINS = {
    "attn_query": 0.5270182885626962,
    "attn_key": 0.5562907139098529,
    "attn_value": 0.3990051254514762,
    "attn_output": 0.3564476277035192,
    "ffn_up": 1.0280206811266708,
    "ffn_down": 0.3048084709039325,
}
COMMON_GAIN = 0.46301170700708616
CHECKPOINT_SHA = "d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00"
DESIGN_SHA = "fc23d52d07d0004afd681116578abdba562dff1a415b059fd72ec261b1e47d70"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class AuditFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def index_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.int64).contiguous()
    array = value.numpy().astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def stable_seed(*items: Any) -> int:
    raw = "|".join(str(item) for item in items).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**63 - 1)


def matrix_key(depth: int, role: str) -> str:
    return f"depth={depth:02d}|role={role}"


def cache_key(panel: str, seed: int, depth: int, role: str) -> str:
    return f"{panel}|seed={seed}|depth={depth:02d}|role={role}"


def exact_grid(include_split: bool, include_arm: bool) -> set[tuple[Any, ...]]:
    values: set[tuple[Any, ...]] = set()
    for panel in PANEL_IDS:
        splits: Sequence[str | None] = SPLITS if include_split else (None,)
        for split in splits:
            for seed in TILING_SEEDS:
                for depth in range(12):
                    for role in ROLES:
                        arms: Sequence[str | None] = ARMS if include_arm else (None,)
                        for arm in arms:
                            row: list[Any] = [panel]
                            if include_split:
                                row.append(split)
                            row.extend((seed, depth, role, matrix_key(depth, role)))
                            if include_arm:
                                row.append(arm)
                            values.add(tuple(row))
    return values


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    require(all(list(row) == fieldnames for row in rows), f"nonuniform CSV rows: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalized_path_is_safe(relative_text: str) -> bool:
    relative = Path(relative_text)
    return bool(
        relative_text
        and not relative.is_absolute()
        and relative_text == relative.as_posix()
        and all(part not in {"", ".", ".."} for part in relative.parts)
        and "data2vec" not in relative_text.casefold()
        and not any(
            part.casefold() in {"target", "targets"}
            or part.casefold().startswith(("target_", "target-"))
            for part in relative.parts
        )
    )


def audit_artifact_manifest(root: Path) -> dict[str, Any]:
    path = root / "artifact_manifest.json"
    require(path.is_file() and not path.is_symlink(), "terminal artifact_manifest.json is absent/symlinked")
    payload = read_json(path)
    require(
        isinstance(payload, Mapping)
        and payload.get("schema_version")
        == "prospective_geometry_matched_replication_artifact_manifest_v1",
        "artifact manifest schema mismatch",
    )
    require(payload.get("manifest_self_excluded") is True, "manifest self-exclusion flag missing")
    records = payload.get("artifacts")
    require(isinstance(records, list) and payload.get("count") == len(records), "manifest count mismatch")
    declared: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(records):
        require(isinstance(row, Mapping) and set(row) == {"path", "sha256", "bytes"}, f"bad manifest row {index}")
        relative = str(row["path"])
        require(normalized_path_is_safe(relative), f"unsafe manifest path: {relative!r}")
        require(relative != "artifact_manifest.json" and relative not in declared, f"duplicate/recursive manifest row: {relative}")
        require(isinstance(row["sha256"], str) and SHA_RE.fullmatch(row["sha256"]) is not None, f"bad SHA: {relative}")
        declared[relative] = row
    actual: set[str] = set()
    for member in root.rglob("*"):
        require(not member.is_symlink(), f"symlink prohibited in formal output: {member}")
        if member.is_file() and member != path:
            relative = member.relative_to(root).as_posix()
            require(normalized_path_is_safe(relative), f"unsafe actual artifact path: {relative!r}")
            actual.add(relative)
    require(actual == set(declared), f"manifest membership mismatch: missing={sorted(actual-set(declared))} stale={sorted(set(declared)-actual)}")
    for relative, row in declared.items():
        member = root / relative
        require(member.stat().st_size == int(row["bytes"]), f"byte count mismatch: {relative}")
        require(sha256_file(member) == row["sha256"], f"file SHA mismatch: {relative}")
    return {
        "pass": True,
        "manifest_sha256": sha256_file(path),
        "file_count": len(records),
        "complete_membership": True,
        "all_hashes_and_bytes_match": True,
        "no_symlinks_or_forbidden_paths": True,
    }


def audit_bindings(root: Path, external_contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    external_bytes = external_contract_path.read_bytes()
    external_sha = hashlib.sha256(external_bytes).hexdigest()
    copied_path = root / "preexecution_contract.json"
    require(copied_path.read_bytes() == external_bytes, "copied contract is not byte-identical")
    contract = json.loads(external_bytes)
    require(contract.get("schema_version") == "prospective_geometry_matched_replication_preexecution_v1", "contract schema mismatch")
    require(contract.get("design_sha256") == DESIGN_SHA, "design SHA mismatch")
    require(contract.get("checkpoint_sha256") == CHECKPOINT_SHA, "checkpoint SHA mismatch")
    binding = read_json(root / "preexecution_binding.json")
    expected_binding = {
        "schema_version": "prospective_geometry_matched_replication_binding_v1",
        "external_contract_path": str(external_contract_path.resolve(strict=True)),
        "external_contract_sha256": external_sha,
        "copied_contract_path": str(copied_path.resolve(strict=True)),
        "copied_contract_sha256": external_sha,
        "design_sha256": DESIGN_SHA,
        "runner_sha256": contract["runner_sha256"],
        "builder_sha256": contract["builder_sha256"],
        "analyzer_sha256": contract["analyzer_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "scripts": contract["scripts"],
        "model_dependencies": contract["model_dependencies"],
        "resources": contract["resources"],
        "panel_cache": contract["panel_cache"],
        "panel_manifest_sha256": contract["panel_cache"]["manifest_sha256"],
    }
    require(binding == expected_binding, "preexecution binding is not an exact contract projection")

    # Hash every exact script/dependency that was eligible to execute.  The
    # audit intentionally does not open panel/model/target resources here.
    hashed = 0
    for section in ("scripts", "model_dependencies"):
        for label, record in contract[section].items():
            path = Path(record["path"])
            require("data2vec" not in str(path).casefold(), f"forbidden dependency path: {path}")
            require(path.is_file() and not path.is_symlink(), f"missing/symlink dependency: {label}")
            require(sha256_file(path) == record["sha256"], f"post-run dependency drift: {label}")
            if "bytes" in record:
                require(path.stat().st_size == int(record["bytes"]), f"dependency byte drift: {label}")
            hashed += 1
    return contract, {
        "pass": True,
        "external_contract_sha256": external_sha,
        "copied_contract_byte_exact": True,
        "binding_exact_projection": True,
        "current_script_and_dependency_hashes_match": True,
        "hashed_script_dependency_files": hashed,
    }


def audit_metadata(root: Path, contract: Mapping[str, Any], external_sha: str) -> dict[str, Any]:
    metadata = read_json(root / "runner_metadata.json")
    required_scalars = {
        "schema_version": "prospective_geometry_matched_replication_runner_v1",
        "design_sha256": DESIGN_SHA,
        "runner_sha256": contract["runner_sha256"],
        "builder_sha256": contract["builder_sha256"],
        "analyzer_sha256": contract["analyzer_sha256"],
        "contract_sha256": external_sha,
        "panel_manifest_sha256": contract["panel_manifest_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "execution_status": "COMPLETE",
        "counts": contract["expected_counts"],
        "decoder_entry_parity_pass": True,
        "known_source_numeric_preflight_pass": True,
        "panel_activations_to_weight_ae": False,
        "full_prediction_matrices_persisted": False,
        "source_gains_applied_by_runner": False,
        "analyzer_invoked": False,
    }
    for key, expected in required_scalars.items():
        require(metadata.get(key) == expected, f"runner metadata drift at {key}")
    require(isinstance(metadata.get("elapsed_seconds"), (int, float)) and metadata["elapsed_seconds"] > 0, "bad elapsed_seconds")
    require(
        metadata.get("target_seal")
        == {
            "installed_before_input_read": True,
            "target_access_events": 0,
            "network_connections": 0,
            "subprocess_launches": 0,
        },
        "terminal target seal is not clean",
    )
    hard = metadata.get("hard_checks")
    expected_hard = {
        "transpose_role_map_parity_pass",
        "raw_stat_algebra_pass",
        "analytic_scale_parity_pass",
        "decoder_entry_parity_pass",
        "A_B_prediction_sha_pass",
        "derangement_pass",
        "tiling_coverage_pass",
        "finite_nondegenerate_pass",
        "source_only_dataflow_pass",
        "loaded_local_module_closure_pass",
    }
    require(isinstance(hard, Mapping) and set(hard) == expected_hard and all(value is True for value in hard.values()), "hard-check grid incomplete")
    expected_panels = [
        {
            "panel_id": panel,
            "path": contract["panel_cache"]["panels"][panel]["path"],
            "sha256": contract["panel_cache"]["panels"][panel]["sha256"],
            "checkpoint_sha256": contract["panel_cache"]["panels"][panel]["checkpoint_sha256"],
            "provenance_pass": True,
            "resource_hash_pass": True,
            "quality_pass": True,
            "validity_pass": True,
        }
        for panel in PANEL_IDS
    ]
    require(metadata.get("panels") == expected_panels, "runner metadata panel binding mismatch")
    require(read_json(root / "resolved_config.json") == contract["execution_config"], "resolved config drift")
    schema = read_json(root / "schema.json")
    require(schema.get("expected_counts") == contract["expected_counts"], "output schema expected-count drift")
    require(schema.get("full_prediction_matrices_persisted") is False, "prediction persistence contract violated")
    require(schema.get("operator_split_predictions_duplicated") is False, "operator prediction duplication flag violated")
    require(schema.get("source_gains_applied") is False, "runner applied gains")
    require(schema.get("analyzer_sha256") == contract["analyzer_sha256"], "schema analyzer SHA drift")
    model = read_json(root / "model_contract.json")
    required_model = {
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_step": 480000,
        "strict_load": True,
        "rope_2d_coord_kind": "raw",
        "use_latent_sampling": False,
        "use_encoder_mu_head": False,
        "disable_z_shortcut": True,
        "flat_lat_dim": 512,
        "patch_size": 16,
        "locked_tile_shape": [64, 64],
        "amp_enabled": True,
        "amp_dtype": "torch.bfloat16",
    }
    require({key: model.get(key) for key in required_model} == required_model, "deterministic raw-RoPE model contract mismatch")
    return {
        "pass": True,
        "execution_status": metadata["execution_status"],
        "elapsed_seconds": metadata["elapsed_seconds"],
        "target_seal_clean": True,
        "raw_rope_deterministic_ae": True,
        "all_hard_checks_declared_true": True,
    }


def audit_preflights(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    known = read_json(root / "known_source_numeric_preflight.json")
    require(known.get("schema_version") == "prospective_known_source_numeric_preflight_v1", "known-source schema mismatch")
    fixed = {
        "pass": True,
        "performed_before_candidate_decode": True,
        "source_activation_passed_to_weight_ae": False,
        "depth": 0,
        "role": "attn_query",
        "tiling_seed": 26081601,
        "code_key": "seed=26081601|depth=0|role=attn_query|enc=cell|code=correct",
        "execution_batch": 64,
    }
    for key, value in fixed.items():
        require(known.get(key) == value, f"known-source preflight drift at {key}")
    require(known.get("cache_file_sha256") == contract["resources"]["source_factorial_code_cache_seed_26081601"]["sha256"], "known-source cache SHA drift")
    require(SHA_RE.fullmatch(str(known.get("partition_sha256", ""))) is not None, "known-source partition SHA malformed")
    comparison = known.get("code_comparison")
    require(isinstance(comparison, Mapping) and comparison.get("tight_bfloat16_aware_pass") is True, "known-source code comparison failed")
    thresholds = comparison.get("fallback_thresholds")
    require(
        thresholds == {"max_abs_lte": 0.0078125, "relative_l2_lte": 0.001, "cosine_gte": 0.999999},
        "known-source fallback thresholds drifted",
    )
    require(
        comparison.get("bit_exact") is True
        or (
            float(comparison["max_abs"]) <= thresholds["max_abs_lte"]
            and float(comparison["relative_l2"]) <= thresholds["relative_l2_lte"]
            and float(comparison["cosine"]) >= thresholds["cosine_gte"]
        ),
        "known-source numeric values do not meet frozen threshold",
    )
    decoder = known.get("decoder_entry_checks")
    require(isinstance(decoder, Mapping) and decoder.get("pass") is True, "known-source decoder parity failed")
    parity = read_json(root / "decoder_entry_parity.json")
    probes = parity.get("probes")
    require(parity.get("pass") is True and isinstance(probes, list) and len(probes) == 4, "decoder parity grid mismatch")
    require({(row.get("role"), int(row.get("execution_batch", -1))) for row in probes} == {(role, batch) for role in ("attn_query", "ffn_up") for batch in (1, 2)}, "decoder parity probe identities mismatch")
    for row in probes:
        require(row.get("pass") is True, f"decoder parity probe failed: {row}")
        for key in (
            "historical_vs_explicit_z_bit_exact",
            "encoder_tokens_actual_vs_none_bit_exact",
            "encoder_tokens_actual_vs_zero_bit_exact",
            "z_shortcut_default_vs_forced_off_bit_exact",
            "z_dec_finite",
            "z_dec_nonzero",
            "all_patch_mask_valid",
        ):
            require(row.get(key) is True, f"decoder parity bit/check failure at {key}")
        require(float(row["historical_vs_explicit_z_max_abs"]) <= 1e-6, "decoder parity max abs too large")
        require(row.get("z_dec_shape") == [int(row["execution_batch"]), 512], "decoder parity latent shape mismatch")
        require((int(row["patch_size"]), int(row["T"]), int(row["d_in_pad"])) == (16, 4, 64), "decoder parity geometry mismatch")

    # Ordering is independently checked from the durable runtime log.
    lines = (root / "run.log").read_text(encoding="utf-8").splitlines()
    require(lines and not any("traceback" in line.casefold() or " | error | " in line.casefold() for line in lines), "runner log contains error/traceback")
    def first(fragment: str) -> int:
        indices = [i for i, line in enumerate(lines) if fragment in line]
        require(bool(indices), f"runner log lacks stage: {fragment}")
        return indices[0]
    require(first("stage=known_source_numeric_preflight_complete") < first("stage=panel_load"), "known-source preflight was not before panel load")
    require(first("stage=weight_ae_ready") < first("stage=known_source_numeric_preflight"), "model/preflight stage order invalid")
    require(first("stage=artifact_write") < first("stage=complete"), "artifact/complete stage order invalid")
    require(sum("stage=prediction_complete" in line for line in lines) == 864, "prediction log count mismatch")
    require(any("rope=raw" in line and "latent_sampling=False" in line for line in lines), "log lacks raw-RoPE deterministic declaration")
    return {
        "pass": True,
        "known_source_code_bit_exact": bool(comparison["bit_exact"]),
        "known_source_max_abs": float(comparison["max_abs"]),
        "known_source_relative_l2": float(comparison["relative_l2"]),
        "known_source_cosine": float(comparison["cosine"]),
        "decoder_parity_probes": len(probes),
        "prediction_log_rows": 864,
        "preflight_preceded_candidate_panel_load": True,
    }


def audit_loaded_modules(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(root / "loaded_local_module_audit.json")
    expected_stages = {"after_helper_imports", "after_model_build", "after_all_weight_ae_forwards"}
    require(isinstance(payload, Mapping) and set(payload) == expected_stages, "loaded-module audit stages mismatch")
    allowed: dict[Path, tuple[str, Mapping[str, Any]]] = {}
    for section in ("scripts", "model_dependencies"):
        for label, record in contract[section].items():
            allowed[Path(record["path"]).resolve(strict=True)] = (f"{section}:{label}", record)
    stage_counts: dict[str, int] = {}
    all_paths: set[Path] = set()
    for stage, report in payload.items():
        require(report.get("pass") is True, f"loaded-module stage failed: {stage}")
        require(int(report.get("allowed_file_count", -1)) == len(allowed), f"allowed file count drift: {stage}")
        rows = report.get("loaded_modules")
        require(isinstance(rows, list) and int(report.get("loaded_module_count", -1)) == len(rows), f"loaded-module count mismatch: {stage}")
        seen_modules: set[str] = set()
        for row in rows:
            require(isinstance(row, Mapping) and set(row) == {"module", "path", "relative_path"}, f"bad loaded-module row: {stage}")
            module = str(row["module"])
            path = Path(row["path"]).resolve(strict=True)
            require(module not in seen_modules, f"duplicate loaded module name: {module}")
            seen_modules.add(module)
            require("data2vec" not in module.casefold() and "data2vec" not in str(path).casefold(), f"forbidden loaded module: {module}")
            require(path in allowed, f"loaded local module escaped closure: {module} -> {path}")
            require(sha256_file(path) == allowed[path][1]["sha256"], f"loaded module hash drift: {module}")
            all_paths.add(path)
        stage_counts[stage] = len(rows)
    require(stage_counts["after_helper_imports"] <= stage_counts["after_model_build"] <= stage_counts["after_all_weight_ae_forwards"], "loaded-module counts are non-monotone")
    return {
        "pass": True,
        "allowed_file_count": len(allowed),
        "unique_loaded_local_paths": len(all_paths),
        "stage_loaded_module_counts": stage_counts,
        "all_loaded_paths_bound_and_hash_exact": True,
        "forbidden_loaded_modules": 0,
    }


def bool_csv(value: str) -> bool:
    require(value in {"True", "False"}, f"noncanonical CSV bool: {value!r}")
    return value == "True"


def int_csv(value: str) -> int:
    parsed = int(value)
    require(str(parsed) == value, f"noncanonical CSV int: {value!r}")
    return parsed


def float_csv(value: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"nonfinite CSV float: {value!r}")
    return parsed


def audit_tilings_and_codes(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    _, tiling_rows = read_csv(root / "tiling_manifest.csv")
    _, code_rows = read_csv(root / "correct_latent_code_manifest.csv")
    _, permutation_rows = read_csv(root / "permutation_manifest.csv")
    require(len(tiling_rows) == 288 and len(code_rows) == 288 and len(permutation_rows) == 5184, "tiling/code/permutation row counts mismatch")

    tiling_payload = torch.load(root / "tiling_indices.pt", map_location="cpu", weights_only=True)
    code_payload = torch.load(root / "correct_latent_codes.pt", map_location="cpu", weights_only=True)
    require(tiling_payload.get("schema_version") == "prospective_tiling_indices_v1", "tiling cache schema mismatch")
    require(code_payload.get("schema_version") == "prospective_correct_latent_codes_v1", "code cache schema mismatch")
    tilings = tiling_payload.get("entries")
    codes = code_payload.get("entries")
    require(isinstance(tilings, Mapping) and len(tilings) == 288, "tiling cache entry mismatch")
    require(isinstance(codes, Mapping) and len(codes) == 288, "code cache entry mismatch")
    expected_cache_keys = {cache_key(panel, seed, depth, role) for panel in PANEL_IDS for seed in TILING_SEEDS for depth in range(12) for role in ROLES}
    require(set(tilings) == expected_cache_keys and set(codes) == expected_cache_keys, "tiling/code cache key grid mismatch")

    tiling_csv = {(r["panel_id"], int_csv(r["tiling_seed"]), int_csv(r["depth"]), r["role"], r["matrix_key"]): r for r in tiling_rows}
    code_csv = {(r["panel_id"], int_csv(r["tiling_seed"]), int_csv(r["depth"]), r["role"], r["matrix_key"]): r for r in code_rows}
    require(set(tiling_csv) == exact_grid(False, False), "tiling CSV exact grid mismatch")
    require(set(code_csv) == exact_grid(False, False), "code CSV exact grid mismatch")
    perm_by_cell: dict[tuple[str, int, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in permutation_rows:
        key = (row["panel"], int_csv(row["tiling_seed"]), int_csv(row["depth"]), row["role"], row["matrix_key"])
        perm_by_cell[key].append(row)
    require(set(perm_by_cell) == exact_grid(False, False), "permutation cell grid mismatch")

    total_tiles = 0
    total_row_derangements = 0
    code_value_min = math.inf
    code_value_max = -math.inf
    code_abs_max = 0.0
    for panel in PANEL_IDS:
        checkpoint = contract["panel_cache"]["panels"][panel]["checkpoint_sha256"]
        for seed in TILING_SEEDS:
            for depth in range(12):
                for role in ROLES:
                    key = matrix_key(depth, role)
                    ckey = cache_key(panel, seed, depth, role)
                    tentry = tilings[ckey]
                    centry = codes[ckey]
                    expected_meta = {"panel_id": panel, "tiling_seed": seed, "depth": depth, "role": role, "matrix_key": key}
                    for field, expected in expected_meta.items():
                        require(tentry.get(field) == expected and centry.get(field) == expected, f"cache metadata drift: {ckey}/{field}")
                    rows = tentry.get("rows")
                    cols = tentry.get("cols")
                    d_in, d_out = ROLE_SHAPES[role]
                    require(isinstance(rows, list) and isinstance(cols, list), f"tiling groups malformed: {ckey}")
                    require(len(rows) == d_in // 64 and len(cols) == d_out // 64, f"tiling group count mismatch: {ckey}")
                    require(all(isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dtype == torch.int64 and tuple(x.shape) == (64,) for x in rows + cols), f"tiling tensor contract mismatch: {ckey}")
                    row_tensor = torch.stack(rows).contiguous()
                    col_tensor = torch.stack(cols).contiguous()
                    require(torch.equal(row_tensor.flatten().sort().values, torch.arange(d_in, dtype=torch.int64)), f"row partition mismatch: {ckey}")
                    require(torch.equal(col_tensor.flatten().sort().values, torch.arange(d_out, dtype=torch.int64)), f"column partition mismatch: {ckey}")
                    for group in row_tensor:
                        require(torch.equal(group, group.sort().values), f"row group unsorted: {ckey}")
                        patch_ids = torch.unique(group // 16)
                        require(len(patch_ids) == 4, f"row group is not four patches: {ckey}")
                        expected_group = torch.cat([torch.arange(int(patch) * 16, int(patch) * 16 + 16, dtype=torch.int64) for patch in patch_ids]).sort().values
                        require(torch.equal(group, expected_group), f"row patch structure mismatch: {ckey}")
                    require(all(torch.equal(group, group.sort().values) for group in col_tensor), f"column group unsorted: {ckey}")

                    # Reproduce the exact frozen random partition from scratch.
                    identity = f"{panel}|{checkpoint}|depth={depth:02d}|role={role}"
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(stable_seed(seed, identity, role, depth))
                    patch_order = torch.randperm(d_in // 16, generator=generator)
                    expected_rows = []
                    offsets = torch.arange(16, dtype=torch.int64)
                    for start in range(0, d_in // 16, 4):
                        patch_group = patch_order[start:start+4].sort().values
                        expected_rows.append((patch_group[:, None] * 16 + offsets[None, :]).reshape(-1).sort().values)
                    col_order = torch.randperm(d_out, generator=generator)
                    expected_cols = [col_order[start:start+64].sort().values for start in range(0, d_out, 64)]
                    require(torch.equal(row_tensor, torch.stack(expected_rows)), f"row tiling RNG reproduction failed: {ckey}")
                    require(torch.equal(col_tensor, torch.stack(expected_cols)), f"column tiling RNG reproduction failed: {ckey}")

                    rsha = index_sha256(row_tensor)
                    csha = index_sha256(col_tensor)
                    psha = hashlib.sha256(f"{rsha}|{csha}".encode("utf-8")).hexdigest()
                    trow = tiling_csv[(panel, seed, depth, role, key)]
                    expected_tiles = (d_in // 64) * (d_out // 64)
                    require(tentry["row_index_sha256"] == rsha == trow["row_index_sha256"], f"row hash mismatch: {ckey}")
                    require(tentry["column_index_sha256"] == csha == trow["column_index_sha256"], f"column hash mismatch: {ckey}")
                    require(tentry["partition_sha256"] == psha == trow["partition_sha256"], f"partition hash mismatch: {ckey}")
                    require((int_csv(trow["row_groups"]), int_csv(trow["column_groups"]), int_csv(trow["num_tiles"])) == (d_in//64, d_out//64, expected_tiles), f"tiling counts mismatch: {ckey}")
                    require((int_csv(trow["coverage_min"]), int_csv(trow["coverage_max"])) == (1, 1), f"tiling coverage mismatch: {ckey}")
                    require(bool_csv(trow["weight_reassembly_bit_exact"]) and bool_csv(trow["coordinate_reassembly_bit_exact"]), f"tiling reassembly failed: {ckey}")

                    tensor = centry.get("codes")
                    require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.float32 and tuple(tensor.shape) == (expected_tiles, 512), f"latent tensor contract mismatch: {ckey}")
                    require(bool(torch.isfinite(tensor).all()), f"latent tensor nonfinite: {ckey}")
                    tsha = tensor_sha256(tensor)
                    crow = code_csv[(panel, seed, depth, role, key)]
                    require(centry.get("codes_shape") == [expected_tiles, 512] and centry.get("codes_dtype") == "torch.float32", f"latent cache declaration mismatch: {ckey}")
                    require(centry.get("tensor_sha256") == tsha == crow["tensor_sha256"], f"latent hash mismatch: {ckey}")
                    require(centry.get("row_index_sha256") == rsha == crow["row_index_sha256"], f"latent row binding mismatch: {ckey}")
                    require(centry.get("column_index_sha256") == csha == crow["column_index_sha256"], f"latent column binding mismatch: {ckey}")
                    require((int_csv(crow["tiles"]), int_csv(crow["code_dim"]), crow["codes_dtype"], crow["cache_key"]) == (expected_tiles, 512, "float32", ckey), f"latent manifest metadata mismatch: {ckey}")
                    code_value_min = min(code_value_min, float(tensor.min()))
                    code_value_max = max(code_value_max, float(tensor.max()))
                    code_abs_max = max(code_abs_max, float(tensor.abs().max()))

                    perm_rows = sorted(perm_by_cell[(panel, seed, depth, role, key)], key=lambda r: int_csv(r["row_group"]))
                    require([int_csv(r["row_group"]) for r in perm_rows] == list(range(d_in // 64)), f"permutation row-group grid mismatch: {ckey}")
                    ncols = d_out // 64
                    for row_group, prow in enumerate(perm_rows):
                        payload = {
                            "depth": depth,
                            "namespace": "within_row_code_derangement_v1",
                            "panel": panel,
                            "role": role,
                            "row_group": row_group,
                            "tiling_seed": seed,
                        }
                        digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
                        offset = 1 + int.from_bytes(digest[:8], "big", signed=False) % (ncols - 1)
                        mapping = (torch.arange(ncols, dtype=torch.int64) + offset) % ncols
                        start = row_group * ncols
                        block = tensor[start:start+ncols]
                        permuted = block[mapping]
                        require({name: prow[name] for name in payload} == {name: str(value) for name, value in payload.items()}, f"permutation payload drift: {ckey}/row={row_group}")
                        require((int_csv(prow["n_col_groups"]), int_csv(prow["offset"]), int_csv(prow["fixed_points"])) == (ncols, offset, 0), f"permutation numeric drift: {ckey}/row={row_group}")
                        require(bool_csv(prow["bijection_pass"]) and bool_csv(prow["same_row_code_multiset_bit_exact"]), f"permutation flags failed: {ckey}/row={row_group}")
                        require(prow["mapping_sha256"] == index_sha256(mapping), f"permutation mapping SHA mismatch: {ckey}/row={row_group}")
                        require(prow["correct_code_block_sha256"] == tensor_sha256(block), f"correct block SHA mismatch: {ckey}/row={row_group}")
                        require(prow["permuted_code_block_sha256"] == tensor_sha256(permuted), f"permuted block SHA mismatch: {ckey}/row={row_group}")
                        require(torch.equal(permuted[mapping.argsort()], block), f"code multiset mismatch: {ckey}/row={row_group}")
                    total_tiles += expected_tiles
                    total_row_derangements += len(perm_rows)
    return {
        "pass": True,
        "tiling_entries": len(tilings),
        "latent_entries": len(codes),
        "latent_tiles_total": total_tiles,
        "permutation_row_groups": total_row_derangements,
        "tilings_rng_reproduced": True,
        "coverage_and_patch_structure_pass": True,
        "permutations_recomputed_from_codes": True,
        "latent_all_finite": True,
        "latent_value_min": code_value_min,
        "latent_value_max": code_value_max,
        "latent_abs_max": code_abs_max,
    }


def parse_stat_rows(rows: list[dict[str, str]], *, operator: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    expected_columns = (
        {"panel_id", "score_split", "tiling_seed", "depth", "role", "matrix_key", "arm", "prediction_sha256", "activation_sha256", "T", "P", "D", "target_energy", "pred_energy", "target_pred_dot"}
        if operator
        else {"panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm", "prediction_sha256", "T", "P", "D", "target_energy", "pred_energy", "target_pred_dot"}
    )
    for index, row in enumerate(rows):
        require(set(row) == expected_columns, f"raw stat CSV columns drift at row {index}")
        converted: dict[str, Any] = dict(row)
        converted["tiling_seed"] = int_csv(row["tiling_seed"])
        converted["depth"] = int_csv(row["depth"])
        for name in ("T", "P", "D", "target_energy", "pred_energy", "target_pred_dot"):
            converted[name] = float_csv(row[name])
        require(converted["T"] == converted["target_energy"] and converted["P"] == converted["pred_energy"] and converted["D"] == converted["target_pred_dot"], f"stat aliases diverged at row {index}")
        require(converted["T"] > 0 and converted["P"] > 0, f"nonpositive T/P at row {index}")
        require(SHA_RE.fullmatch(row["prediction_sha256"]) is not None, f"bad prediction SHA at row {index}")
        if operator:
            require(SHA_RE.fullmatch(row["activation_sha256"]) is not None, f"bad activation SHA at row {index}")
        output.append(converted)
    return output


def metric(T: float, P: float, D: float, gain: float) -> dict[str, float]:
    scaled_p = gain * gain * P
    scaled_d = gain * D
    cosine = scaled_d / math.sqrt(T * scaled_p)
    radius = math.sqrt(scaled_p / T)
    error = (T - 2.0 * scaled_d + scaled_p) / T
    angular = 1.0 - cosine * cosine
    radial = (radius - cosine) ** 2
    require(all(math.isfinite(value) for value in (cosine, radius, error, angular, radial)), "nonfinite derived metric")
    require(abs(error - (angular + radial)) <= 5e-12 * max(1.0, abs(error)), "radial/angular identity failure")
    return {"E": error, "cosine": cosine, "radius": radius, "angular_floor": angular, "radial_penalty": radial}


def aggregate_stats(rows: list[dict[str, Any]], *, operator: bool) -> list[dict[str, Any]]:
    dimensions = ["panel_id"]
    if operator:
        dimensions.append("score_split")
    dimensions.extend(("tiling_seed", "arm"))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    expected_groups = len(PANEL_IDS) * len(TILING_SEEDS) * len(ARMS) * (len(SPLITS) if operator else 1)
    require(len(grouped) == expected_groups and all(len(group) == 72 for group in grouped.values()), "aggregate input groups incomplete")
    output: list[dict[str, Any]] = []
    for group_key, group in sorted(grouped.items()):
        role_stats: dict[str, tuple[float, float, float]] = {}
        for role in ROLES:
            role_rows = [row for row in group if row["role"] == role]
            require(len(role_rows) == 12, "role/depth aggregation incomplete")
            role_stats[role] = tuple(math.fsum(float(row[name]) for row in role_rows) for name in ("T", "P", "D"))
        for calibration in ("raw", "source_role", "source_common"):
            role_metrics: dict[str, dict[str, float]] = {}
            for role, (T, P, D) in role_stats.items():
                gain = 1.0 if calibration == "raw" else ROLE_GAINS[role] if calibration == "source_role" else COMMON_GAIN
                role_metrics[role] = metric(T, P, D, gain)
            macro = {name: math.fsum(role_metrics[role][name] for role in ROLES) / len(ROLES) for name in ("E", "cosine", "radius", "angular_floor", "radial_penalty")}
            T_micro = math.fsum(role_stats[role][0] for role in ROLES)
            P_micro = 0.0
            D_micro = 0.0
            for role, (_T, P, D) in role_stats.items():
                gain = 1.0 if calibration == "raw" else ROLE_GAINS[role] if calibration == "source_role" else COMMON_GAIN
                P_micro += gain * gain * P
                D_micro += gain * D
            micro = metric(T_micro, P_micro, D_micro, 1.0)
            prefix: dict[str, Any] = dict(zip(dimensions, group_key, strict=True))
            for aggregation, values in (("macro", macro), ("micro", micro)):
                output.append({**prefix, "calibration": calibration, "aggregation": aggregation, **values})
    return output


def audit_stats(
    root: Path,
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    _, raw_weight = read_csv(root / "weight_sufficient_stats.csv")
    _, raw_operator = read_csv(root / "operator_sufficient_stats.csv")
    _, predictions = read_csv(root / "prediction_manifest.csv")
    weights = parse_stat_rows(raw_weight, operator=False)
    operators = parse_stat_rows(raw_operator, operator=True)
    require(len(weights) == 864 and len(operators) == 1728 and len(predictions) == 864, "raw statistic/prediction counts mismatch")
    weight_keys = {(r["panel_id"], r["tiling_seed"], r["depth"], r["role"], r["matrix_key"], r["arm"]) for r in weights}
    operator_keys = {(r["panel_id"], r["score_split"], r["tiling_seed"], r["depth"], r["role"], r["matrix_key"], r["arm"]) for r in operators}
    require(weight_keys == exact_grid(False, True), "weight sufficient-stat exact grid mismatch")
    require(operator_keys == exact_grid(True, True), "operator sufficient-stat exact grid mismatch")
    pred_map: dict[tuple[Any, ...], dict[str, str]] = {}
    for index, row in enumerate(predictions):
        require(set(row) == {"panel_id", "tiling_seed", "depth", "role", "matrix_key", "arm", "prediction_sha256", "prediction_shape", "prediction_persisted", "score_split_reuse_count"}, f"prediction manifest columns drift at {index}")
        key = (row["panel_id"], int_csv(row["tiling_seed"]), int_csv(row["depth"]), row["role"], row["matrix_key"], row["arm"])
        require(key not in pred_map, f"duplicate prediction manifest key: {key}")
        role = row["role"]
        require(row["prediction_shape"] == f"{ROLE_SHAPES[role][0]}x{ROLE_SHAPES[role][1]}", f"prediction shape drift: {key}")
        require(row["prediction_persisted"] == "False" and int_csv(row["score_split_reuse_count"]) == 2, f"prediction persistence/reuse drift: {key}")
        require(SHA_RE.fullmatch(row["prediction_sha256"]) is not None, f"prediction SHA malformed: {key}")
        pred_map[key] = row
    require(set(pred_map) == exact_grid(False, True), "prediction manifest exact grid mismatch")

    weight_by_key = {(r["panel_id"], r["tiling_seed"], r["depth"], r["role"], r["matrix_key"], r["arm"]): r for r in weights}
    operator_by_key = {(r["panel_id"], r["score_split"], r["tiling_seed"], r["depth"], r["role"], r["matrix_key"], r["arm"]): r for r in operators}
    max_cauchy_violation = 0.0
    min_cauchy_slack = math.inf
    diagnostics: list[dict[str, Any]] = []
    distribution_summary: list[dict[str, Any]] = []
    for space, rows in (("weight", weights), ("operator", operators)):
        for row in rows:
            denominator = row["T"] * row["P"]
            relative_violation = max(0.0, (row["D"] * row["D"] - denominator) / denominator)
            relative_slack = (denominator - row["D"] * row["D"]) / denominator
            max_cauchy_violation = max(max_cauchy_violation, relative_violation)
            min_cauchy_slack = min(min_cauchy_slack, relative_slack)
            require(relative_violation <= 2e-12, f"Cauchy violation in {space}: {row}")
            raw_metric = metric(row["T"], row["P"], row["D"], 1.0)
            calibrated_metric = metric(row["T"], row["P"], row["D"], ROLE_GAINS[row["role"]])
            reasons: list[str] = []
            if raw_metric["E"] > 2.0:
                reasons.append("raw_E_gt_2")
            if raw_metric["radius"] > 1.5:
                reasons.append("raw_radius_gt_1p5")
            if row["arm"] == "correct" and raw_metric["cosine"] < 0.0:
                reasons.append("correct_negative_cosine")
            if row["arm"] == "correct" and calibrated_metric["E"] >= 1.0:
                reasons.append("correct_source_role_E_ge_1")
            if calibrated_metric["E"] > 2.0:
                reasons.append("source_role_E_gt_2")
            if reasons:
                diagnostics.append(
                    {
                        "space": space,
                        "panel_id": row["panel_id"],
                        "score_split": row.get("score_split", ""),
                        "tiling_seed": row["tiling_seed"],
                        "depth": row["depth"],
                        "role": row["role"],
                        "matrix_key": row["matrix_key"],
                        "arm": row["arm"],
                        "diagnostic_reasons": ";".join(reasons),
                        "raw_E": raw_metric["E"],
                        "source_role_E": calibrated_metric["E"],
                        "raw_cosine": raw_metric["cosine"],
                        "raw_radius": raw_metric["radius"],
                        "source_role_radius": calibrated_metric["radius"],
                        "relative_cauchy_slack": relative_slack,
                    }
                )

        for arm in ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            raw_values = [metric(row["T"], row["P"], row["D"], 1.0) for row in arm_rows]
            calibrated_values = [
                metric(row["T"], row["P"], row["D"], ROLE_GAINS[row["role"]])
                for row in arm_rows
            ]
            summary: dict[str, Any] = {"space": space, "arm": arm, "rows": len(arm_rows)}
            for label, values, field in (
                ("raw_E", raw_values, "E"),
                ("source_role_E", calibrated_values, "E"),
                ("raw_cosine", raw_values, "cosine"),
                ("raw_radius", raw_values, "radius"),
            ):
                ordered = sorted(float(item[field]) for item in values)
                summary[f"{label}_min"] = ordered[0]
                summary[f"{label}_median"] = ordered[len(ordered) // 2]
                summary[f"{label}_p95"] = ordered[int(0.95 * (len(ordered) - 1))]
                summary[f"{label}_max"] = ordered[-1]
            distribution_summary.append(summary)

    # Exact prediction SHA binding and target invariants.
    for key, pred in pred_map.items():
        require(weight_by_key[key]["prediction_sha256"] == pred["prediction_sha256"], f"weight/prediction SHA mismatch: {key}")
        split_hashes = []
        for split in SPLITS:
            op_key = (key[0], split, *key[1:])
            op = operator_by_key[op_key]
            require(op["prediction_sha256"] == pred["prediction_sha256"], f"operator/prediction SHA mismatch: {op_key}")
            expected_activation = contract["panel_cache"]["panels"][key[0]]["matrix_audit"][key[4]]["activation_tensor_sha256"][split]
            require(op["activation_sha256"] == expected_activation, f"operator activation SHA mismatch: {op_key}")
            split_hashes.append(op["prediction_sha256"])
        require(len(set(split_hashes)) == 1, f"A/B prediction reuse failed: {key}")
    for panel in PANEL_IDS:
        for seed in TILING_SEEDS:
            for depth in range(12):
                for role in ROLES:
                    key = matrix_key(depth, role)
                    arm_hashes = {pred_map[(panel, seed, depth, role, key, arm)]["prediction_sha256"] for arm in ARMS}
                    require(len(arm_hashes) == 3, f"arm prediction SHA collision: {panel}/{seed}/{key}")
                    weight_t = {weight_by_key[(panel, current_seed, depth, role, key, arm)]["T"] for current_seed in TILING_SEEDS for arm in ARMS}
                    require(len(weight_t) == 1, f"weight target T drift over arm/tiling: {panel}/{key}")
                    for split in SPLITS:
                        operator_t = {operator_by_key[(panel, split, current_seed, depth, role, key, arm)]["T"] for current_seed in TILING_SEEDS for arm in ARMS}
                        require(len(operator_t) == 1, f"operator target T drift over arm/tiling: {panel}/{split}/{key}")
                        activation_hashes = {operator_by_key[(panel, split, current_seed, depth, role, key, arm)]["activation_sha256"] for current_seed in TILING_SEEDS for arm in ARMS}
                        require(len(activation_hashes) == 1, f"activation SHA drift over arm/tiling: {panel}/{split}/{key}")

    weight_aggregates = aggregate_stats(weights, operator=False)
    operator_aggregates = aggregate_stats(operators, operator=True)
    lookup = {(r["panel_id"], r["score_split"], r["tiling_seed"], r["aggregation"], r["calibration"], r["arm"]): r for r in operator_aggregates}
    effects: list[dict[str, Any]] = []
    for panel in PANEL_IDS:
        for split in SPLITS:
            for seed in TILING_SEEDS:
                for aggregation in ("macro", "micro"):
                    for control in ("permuted_within_row", "zero_code"):
                        correct = lookup[(panel, split, seed, aggregation, "source_role", "correct")]
                        comparator = lookup[(panel, split, seed, aggregation, "source_role", control)]
                        effects.append({
                            "panel_id": panel,
                            "score_split": split,
                            "tiling_seed": seed,
                            "aggregation": aggregation,
                            "control": control,
                            "correct_E_source_role": correct["E"],
                            "control_E_source_role": comparator["E"],
                            "control_over_correct_E_ratio": comparator["E"] / correct["E"],
                            "correct_cosine_source_role": correct["cosine"],
                            "control_cosine_source_role": comparator["cosine"],
                            "correct_minus_control_cosine": correct["cosine"] - comparator["cosine"],
                        })
    effect_ratios = [float(r["control_over_correct_E_ratio"]) for r in effects]
    cosine_effects = [float(r["correct_minus_control_cosine"]) for r in effects]
    diagnostic_reason_counts: Counter[str] = Counter()
    for row in diagnostics:
        diagnostic_reason_counts.update(row["diagnostic_reasons"].split(";"))
    return (
        {
            "pass": True,
            "weight_rows": len(weights),
            "operator_rows": len(operators),
            "prediction_rows": len(predictions),
            "all_exact_grids_complete": True,
            "all_stats_finite_TP_positive": True,
            "max_relative_cauchy_violation": max_cauchy_violation,
            "min_relative_cauchy_slack": min_cauchy_slack,
            "A_B_prediction_reuse_pass": True,
            "prediction_arm_collision_cells": 0,
            "target_and_activation_invariants_pass": True,
            "diagnostic_rows_flagged": len(diagnostics),
            "diagnostic_reason_counts": dict(sorted(diagnostic_reason_counts.items())),
            "source_role_effect_ratio_min": min(effect_ratios),
            "source_role_effect_ratio_max": max(effect_ratios),
            "source_role_cosine_effect_min": min(cosine_effects),
            "source_role_cosine_effect_max": max(cosine_effects),
        },
        weight_aggregates,
        operator_aggregates,
        effects,
        diagnostics,
        distribution_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-output", type=Path, required=True)
    parser.add_argument("--external-contract", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.runner_output.resolve(strict=True)
    external = args.external_contract.resolve(strict=True)
    audit_output = args.audit_output.resolve(strict=False)
    for value in (root, external, audit_output):
        require("data2vec" not in str(value).casefold(), f"forbidden input/output path: {value}")
    require(root.is_dir() and not root.is_symlink(), f"formal output is not a real directory: {root}")
    require(not audit_output.exists(), f"audit output must be fresh: {audit_output}")
    require(audit_output != root and not audit_output.is_relative_to(root), "audit output cannot modify formal output")

    manifest_audit = audit_artifact_manifest(root)
    contract, binding_audit = audit_bindings(root, external)
    metadata_audit = audit_metadata(root, contract, binding_audit["external_contract_sha256"])
    preflight_audit = audit_preflights(root, contract)
    module_audit = audit_loaded_modules(root, contract)
    tiling_audit = audit_tilings_and_codes(root, contract)
    (
        stats_audit,
        weight_aggregates,
        operator_aggregates,
        effects,
        raw_diagnostics,
        raw_distribution_summary,
    ) = audit_stats(root, contract)

    audit_output.mkdir(parents=True, exist_ok=False)
    write_csv(audit_output / "independent_weight_point_aggregates.csv", weight_aggregates)
    write_csv(audit_output / "independent_operator_point_aggregates.csv", operator_aggregates)
    write_csv(audit_output / "independent_source_role_directional_effects.csv", effects)
    write_csv(audit_output / "raw_stat_diagnostic_rows.csv", raw_diagnostics)
    write_csv(audit_output / "raw_stat_distribution_summary.csv", raw_distribution_summary)
    report = {
        "schema_version": "prospective_raw_runner_independent_audit_v1",
        "scope": "raw formal runner only; CPU analyzer not executed",
        "runner_output": str(root),
        "runner_artifact_manifest_sha256": manifest_audit["manifest_sha256"],
        "external_contract": str(external),
        "external_contract_sha256": binding_audit["external_contract_sha256"],
        "checks": {
            "artifact_manifest": manifest_audit,
            "contract_binding": binding_audit,
            "runner_metadata": metadata_audit,
            "known_source_and_decoder_preflight": preflight_audit,
            "loaded_local_module_closure": module_audit,
            "tilings_latents_permutations": tiling_audit,
            "raw_statistics_and_point_effects": stats_audit,
        },
        "verdict_for_cpu_analyzer": "GO",
        "limitations": [
            "This audit does not execute the frozen CPU analyzer or bootstrap.",
            "Point effects reconstructed here are validation evidence, not the final criterion verdict.",
            "Full decoded predictions were intentionally not persisted, so T/P/D cannot be recomputed from matrices post hoc.",
        ],
    }
    report_path = audit_output / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated = [path for path in audit_output.iterdir() if path.is_file()]
    manifest = {
        "schema_version": "prospective_raw_runner_independent_audit_manifest_v1",
        "files": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(generated)
        ],
    }
    (audit_output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
