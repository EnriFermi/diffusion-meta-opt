#!/usr/bin/env python3
"""Source-only C_enc x W-code x C_dec causal factorial.

This diagnostic reuses the exact 72-matrix held-out ViT-B panel, A/B activation
references, source templates, and two locked tilings written by
``source_confirmatory_g01_gate.py``.  It has no target-domain path or target
model option.

The intervention point is the deterministic decoder code

    z_dec = latent_norm(encoder_latent_slots.flatten(1))

rather than raw latent slots.  At the canonical checkpoint the normal decoder
uses latent KV, no encoder-token query hint, no direct encoder-token path, and
has the z shortcut disabled.  Therefore replacing ``z_dec`` removes every
W-dependent decoder input.  The runtime preflight verifies these statements
against the loaded model before the full factorial can run.

The script is audit-only unless ``--execute`` is supplied.  Audit-only mode
does not load the Weight-AE or call the distribution encoder.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import source_confirmatory_g01_gate as gate  # noqa: E402


DEFAULT_PARENT_RUN = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_confirmatory_gate_20260816_clean2"
)
DEFAULT_OUTPUT = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_latent_code_factorial_20260816"
)
EXPECTED_PARENT_REALPATH = Path(
    "/home/coder/project/projects/shared/storage/artifacts/crossmodal_united_structure/"
    "source_confirmatory_gate_20260816_clean2"
)
EXPECTED_HELDOUT_REALPATH = Path(
    "/home/coder/project/projects/weight-vae/workspace/post_train_research/"
    "big_vae_heldout_eval/artifacts/offline_dataset"
)
EXPECTED_CHECKPOINT_REALPATH = Path(
    "/home/coder/project/projects/shared/storage/artifacts/training/checkpoints/"
    "weight_quantile_vae_gpu0_square/stage_1/latest.pt"
)
EXPECTED_OUTPUT_ROOT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/crossmodal_united_structure"
)
POSTRUN_ANALYZER = Path(__file__).resolve().with_name("analyze_source_latent_code_factorial.py")

# Immutable clean2 reference.  A user-supplied parent directory is accepted by
# argparse for an explicit error message, not as a scientifically valid override.
EXPECTED_PARENT_HASHES = {
    "heldout_panel_manifest.json": "10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985",
    "tiling_coverage.csv": "876658595aec5f42736ae995f5c45d74f4d76a744aaaac9819f23cc9cc0f17b5",
    "source_condition_templates.pt": "8fc6c61bb6baae4e7b1d618133ec651a91386d90c66f540182faa7dfb1655f99",
    "condition_template_summary.csv": "abcffb2d260ee0eed51c251331919c8d36ba7e8bd04e91d68c372a19c338c6b5",
    "matrix_metrics.csv": "d9d42484c55bd74a51da773edd3596306ad7f4b7d7f6055da90e1f284e7d19a1",
    "aggregate_metrics.csv": "85270f56526ba56a721c863223f1b230da4db22e093e9b8335d98e1722338c51",
    "block_bootstrap.csv": "233058de4df5518f5154ddcd6b755ecd69d1e73896f59d1c2ea84919380b66f9",
    "checkpoint_info.json": "5093fe3d19e8e6ca112f308ff77bf1cf812b687e54f13e73100fbda6265cdf30",
    "run_manifest.json": "853fff83debf96e778c50ef2cc33ddb88c06632258bed0af1daf00fae89f2b6a",
    "resolved_config.json": "b374b86e1b541f7107cbe448ea6bd9b91e9d82152b356c053ba799c5d02995b9",
    "execution_manifest.json": "9114f3b1942231b3a933d224029f2718a89fb9e442b645cbe4aee0ceb07af31f",
    "decisions.json": "477a51d7021255055ecbb8e132468918a041d6d3583c2e10f97ca170d1f98bbc",
}
EXPECTED_DEPENDENCY_HASHES = {
    "experiments/source_confirmatory_g01_gate.py": "e00da48c45fb94d443b818010951dc21a8b69f1f1d0dac37b419368a47b625f4",
    "big_vae/models/big_weight_vae_parts/encoding_mixin.py": "1171c75b1dfc5c936243b1d51a7d796b0195ea2ca5e6cddbb2efca2affd266c5",
    "big_vae/models/big_weight_vae_parts/decoding_mixin.py": "d5e7ef2897ae66b83ea4649692685d119e6eae633fcefd80f4f8436c23a269df",
    "big_vae/models/big_weight_vae_parts/latent_mixin.py": "7f04d15c1cd5ea39b9bca809c99e38dc0a055ded95a4897a728dae8b635a2a26",
    "big_vae/models/big_weight_vae_parts/config.py": "7385b795eda5c0624f438d6aced325d297dc9a3479d25f80799845ac1e3e0c56",
    "big_vae/models/big_weight_vae_parts/core.py": "3e7bd09d5f78575d1106fd7136ca206cd009d6d6ee837497d84582730a694ec8",
    "training/big_vae/model_config.py": "31a63b33de61533c9e93f72041f9030f2ed2152430cf0293b918887f2971d427",
    "training/big_vae/runtime.py": "4723aaca95be7ce32e4effe4e894b929e110dc11b44a925eb4308f846ec1c0c0",
    "training/big_vae/checkpointing.py": "5b6562832b0eef241fbf4b5dfb96a39da1366da7f5cdc10fc8f0b00f98894bdd",
}

ENC_CONDITIONS = ("cell", "native")
CODE_CONDITIONS = (
    "correct",
    "permuted_within_row",
    "deranged_block",
    "zero",
)
DEC_CONDITIONS = (
    "cell",
    "native",
    "wrong_role_same_depth",
    "same_role_wrong_depth",
)
CODE_WITH_ENCODER = ("correct", "permuted_within_row", "deranged_block")

REQUIRED_PARENT_FILES = (
    "aggregate_metrics.csv",
    "block_bootstrap.csv",
    "checkpoint_info.json",
    "condition_template_summary.csv",
    "decisions.json",
    "execution_manifest.json",
    "heldout_panel_manifest.json",
    "matrix_metrics.csv",
    "resolved_config.json",
    "run_manifest.json",
    "source_condition_templates.pt",
    "tiling_coverage.csv",
)

TARGET_ACCESS_SEAL: dict[str, Any] = {
    "installed": False,
    "forbidden_path_markers": ["data2vec"],
    "network_connect_prohibited": True,
}


@dataclass(frozen=True)
class Arm:
    encoder_condition: str
    code_condition: str
    decoder_condition: str

    @property
    def key(self) -> str:
        return f"enc={self.encoder_condition}|code={self.code_condition}|dec={self.decoder_condition}"


COMPUTED_ARMS = tuple(
    Arm(enc, code, dec)
    for enc in ENC_CONDITIONS
    for code in CODE_WITH_ENCODER
    for dec in DEC_CONDITIONS
) + tuple(Arm("none", "zero", dec) for dec in DEC_CONDITIONS)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_RUN)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--bootstrap-draws", type=int, default=10_000)
    value.add_argument("--factorial-seed", type=int, default=26_081_677)
    value.add_argument("--no-amp", action="store_true")
    value.add_argument(
        "--persist-code-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist z_dec and native C_patch tensors for audit/reuse (roughly 250 MiB total).",
    )
    value.add_argument(
        "--execute",
        action="store_true",
        help="Actually load the Weight-AE and run the source-only factorial.",
    )
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_no_target_modules() -> None:
    forbidden = sorted(name for name in sys.modules if "data2vec" in name.lower())
    if forbidden:
        raise RuntimeError(f"target/data2vec module imported in source-only process: {forbidden}")


def install_target_access_seal() -> dict[str, Any]:
    """Install a process-wide executable block on target paths and network reads."""
    if TARGET_ACCESS_SEAL["installed"]:
        return dict(TARGET_ACCESS_SEAL)

    forbidden_markers = tuple(str(value).lower() for value in TARGET_ACCESS_SEAL["forbidden_path_markers"])

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir"} and args:
            candidate = args[0]
            if isinstance(candidate, (str, bytes, Path)):
                normalized = str(candidate).lower()
                if any(marker in normalized for marker in forbidden_markers):
                    raise RuntimeError(f"source-only target seal blocked filesystem event={event}: {candidate}")
        if event == "socket.connect":
            raise RuntimeError("source-only target seal blocked a network connection")
        if event == "subprocess.Popen" and args:
            normalized = " ".join(str(value) for value in args).lower()
            if any(marker in normalized for marker in forbidden_markers):
                raise RuntimeError("source-only target seal blocked a target-related subprocess")

    sys.addaudithook(audit_hook)
    TARGET_ACCESS_SEAL["installed"] = True
    TARGET_ACCESS_SEAL["audit_events"] = ["open", "os.listdir", "os.scandir", "socket.connect", "subprocess.Popen"]
    assert_no_target_modules()
    return dict(TARGET_ACCESS_SEAL)


def require_exact_realpath(path: Path, expected: Path, *, label: str) -> Path:
    actual = path.resolve(strict=True)
    expected_value = expected.resolve(strict=True)
    if actual != expected_value:
        raise RuntimeError(f"{label} realpath mismatch: actual={actual} expected={expected_value}")
    return actual


def assert_output_allowlisted(output: Path) -> Path:
    output_value = output.resolve()
    allowed_root = EXPECTED_OUTPUT_ROOT.resolve(strict=True)
    if not output_value.is_relative_to(allowed_root) or output_value == allowed_root:
        raise RuntimeError(f"output must be a child of the source diagnostic root: {output_value}")
    if output_value.is_relative_to(EXPECTED_HELDOUT_REALPATH.resolve(strict=True)):
        raise RuntimeError(f"output overlaps immutable heldout source bank: {output_value}")
    return output_value


def audit_code_dependencies() -> dict[str, str]:
    actual: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in EXPECTED_DEPENDENCY_HASHES.items():
        path = WORKSPACE / relative
        if not path.is_file():
            raise FileNotFoundError(f"required transitive code dependency missing: {path}")
        digest = gate.sha256_file(path)
        actual[relative] = digest
        if digest != expected:
            mismatches[relative] = {"actual": digest, "expected": expected}
    if mismatches:
        raise RuntimeError(f"transitive evaluator/model dependency changed: {mismatches}")
    return actual


def recheck_parent_immutable(parent: Path, audit: dict[str, Any]) -> dict[str, str]:
    actual = {name: gate.sha256_file(parent / name) for name in REQUIRED_PARENT_FILES}
    if actual != audit["parent_hashes"] or any(
        actual[name] != expected for name, expected in EXPECTED_PARENT_HASHES.items()
    ):
        raise RuntimeError("immutable parent artifacts changed after preexecution audit")
    assert_no_target_modules()
    return actual


def ref_from_dict(value: dict[str, Any]) -> gate.RecordRef:
    return gate.RecordRef(
        chunk_idx=int(value["chunk_idx"]),
        record_idx=int(value["record_idx"]),
        source_key=str(value["source_key"]),
        model_name=str(value["model_name"]),
        layer_name=str(value["layer_name"]),
        role=str(value["role"]),
        depth=int(value["depth"]),
        weight_shape=tuple(int(item) for item in value["weight_shape"]),
        primary_dataset=str(value["primary_dataset"]),
    )


def csv_sha_index(rows: Iterable[dict[str, str]]) -> dict[tuple[str, int, int], str]:
    return {
        # The immutable clean2 tiling artifact uses the concrete ViT block
        # column name ``depth`` (not ``canonical_depth``).
        (str(row["role"]), int(row["depth"]), int(row["tiling_seed"])): str(
            row["partition_sha256"]
        )
        for row in rows
    }


def wrong_condition_map() -> list[dict[str, Any]]:
    """Separate balanced role-only and depth-only decoder-condition maps."""
    rows: list[dict[str, Any]] = []
    for role_idx, role in enumerate(gate.ROLES):
        wrong_role = gate.ROLES[(role_idx + 1) % len(gate.ROLES)]
        for depth in range(12):
            rows.extend(
                [
                    {
                        "decoder_condition": "wrong_role_same_depth",
                        "target_role": role,
                        "target_depth": depth,
                        "template_role": wrong_role,
                        "template_depth": depth,
                        "rule": "role cyclic +1; depth fixed; bijective and no fixed role",
                    },
                    {
                        "decoder_condition": "same_role_wrong_depth",
                        "target_role": role,
                        "target_depth": depth,
                        "template_role": role,
                        "template_depth": (depth + 6) % 12,
                        "rule": "role fixed; depth cyclic +6; bijective and no fixed depth",
                    },
                ]
            )
    return rows


def donor_block_map() -> list[dict[str, Any]]:
    """Same-role, maximally depth-separated, balanced donor mapping."""
    return [
        {
            "role": role,
            "target_depth": depth,
            "donor_depth": (depth + 6) % 12,
            "rule": (
                "donor W is retiled with the target locked tiling and re-encoded under the target C_enc; "
                "only W changes"
            ),
        }
        for role in gate.ROLES
        for depth in range(12)
    ]


def arm_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for enc in ENC_CONDITIONS:
        for code in CODE_CONDITIONS:
            for dec in DEC_CONDITIONS:
                rows.append(
                    {
                        "encoder_condition": enc,
                        "code_condition": code,
                        "decoder_condition": dec,
                        "computed_arm": (
                            Arm("none", "zero", dec).key
                            if code == "zero"
                            else Arm(enc, code, dec).key
                        ),
                        "zero_code_reused_across_encoder_labels": code == "zero",
                    }
                )
    return rows


def audit_parent(parent: Path) -> dict[str, Any]:
    assert_no_target_modules()
    parent = require_exact_realpath(parent, EXPECTED_PARENT_REALPATH, label="parent run")
    missing = [str(parent / name) for name in REQUIRED_PARENT_FILES if not (parent / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing parent artifacts: {missing}")

    run_manifest = read_json(parent / "run_manifest.json")
    execution = read_json(parent / "execution_manifest.json")
    resolved = read_json(parent / "resolved_config.json")
    checkpoint = read_json(parent / "checkpoint_info.json")
    panel = read_json(parent / "heldout_panel_manifest.json")
    tiling_rows = read_csv(parent / "tiling_coverage.csv")
    required_tiling_columns = {
        "tiling_seed",
        "depth",
        "role",
        "source_key",
        "shape",
        "num_tiles",
        "partition_sha256",
    }
    actual_tiling_columns = set(tiling_rows[0]) if tiling_rows else set()
    if not required_tiling_columns.issubset(actual_tiling_columns) or "canonical_depth" in actual_tiling_columns:
        raise RuntimeError(
            "parent tiling CSV schema mismatch: "
            f"required={sorted(required_tiling_columns)} actual={sorted(actual_tiling_columns)}"
        )
    parent_hashes = {name: gate.sha256_file(parent / name) for name in REQUIRED_PARENT_FILES}
    hash_mismatches = {
        name: {"actual": parent_hashes.get(name), "expected": expected}
        for name, expected in EXPECTED_PARENT_HASHES.items()
        if parent_hashes.get(name) != expected
    }
    if hash_mismatches:
        raise RuntimeError(f"immutable clean2 parent hash mismatch: {hash_mismatches}")

    if str(run_manifest.get("status")) != "COMPLETE_G0_G1":
        raise RuntimeError(f"parent run is not complete: {run_manifest.get('status')}")
    if bool(run_manifest.get("target_data2vec_access")):
        raise RuntimeError("parent run reports target access; source-only reuse is prohibited")
    if str(execution.get("target_data2vec_access")) != "PROHIBITED_AND_NOT_PERFORMED":
        raise RuntimeError("parent target-seal contract is absent")
    if int(run_manifest["counts"]["heldout_matrices"]) != 72 or len(panel) != 72:
        raise RuntimeError("parent held-out panel is not the exact 72-matrix panel")
    if checkpoint.get("sha256") != gate.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"wrong parent checkpoint: {checkpoint}")
    if int(checkpoint.get("step", -1)) != gate.EXPECTED_CHECKPOINT_STEP:
        raise RuntimeError(f"wrong parent checkpoint step: {checkpoint}")
    if checkpoint.get("rope_2d_coord_kind") != "raw" or checkpoint.get("use_latent_sampling") is not False:
        raise RuntimeError(f"parent deterministic-AE contract failed: {checkpoint}")
    if checkpoint.get("strict_load") is not True:
        raise RuntimeError(f"parent checkpoint was not strict-loaded: {checkpoint}")
    checkpoint_realpath = require_exact_realpath(
        Path(checkpoint["path"]), EXPECTED_CHECKPOINT_REALPATH, label="canonical checkpoint"
    )
    checkpoint_file_sha256 = gate.sha256_file(checkpoint_realpath)
    if checkpoint_file_sha256 != gate.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "canonical checkpoint bytes changed: "
            f"actual={checkpoint_file_sha256} expected={gate.EXPECTED_CHECKPOINT_SHA256}"
        )
    heldout_realpath = require_exact_realpath(
        Path(resolved["heldout_root"]), EXPECTED_HELDOUT_REALPATH, label="heldout source bank"
    )
    panel_models = {
        str(ref["model_name"])
        for item in panel
        for ref in (item["context_a"], item["score_b"])
    }
    if panel_models != {"vit_base_p16_224"}:
        raise RuntimeError(f"heldout panel model mismatch: {panel_models}")
    if any(
        str(item[side]["source_key"]) == ""
        or str(item[side]["primary_dataset"]) != "flickr30k"
        for item in panel
        for side in ("context_a", "score_b")
    ):
        raise RuntimeError("heldout panel contains an empty source key or a non-flickr30k record")
    if any(
        (int(item["context_a"]["chunk_idx"]), int(item["context_a"]["record_idx"]))
        == (int(item["score_b"]["chunk_idx"]), int(item["score_b"]["record_idx"]))
        for item in panel
    ):
        raise RuntimeError("heldout context A and score B are not disjoint")
    if len(tiling_rows) != 144:
        raise RuntimeError(f"parent tiling rows must be 144, got {len(tiling_rows)}")
    tiling_index = csv_sha_index(tiling_rows)
    expected_tiling_keys = {
        (role, depth, seed)
        for seed in (26_081_601, 26_081_602)
        for depth in range(12)
        for role in gate.ROLES
    }
    if len(tiling_index) != len(tiling_rows) or set(tiling_index) != expected_tiling_keys:
        raise RuntimeError("parent tiling CSV has duplicate or missing (role,depth,seed) rows")
    seeds = tuple(int(value) for value in resolved["tiling_seeds"])
    if seeds != (26_081_601, 26_081_602):
        raise RuntimeError(f"expected two parent tiling seeds, got {seeds}")
    if sorted({int(row["tiling_seed"]) for row in tiling_rows}) != sorted(seeds):
        raise RuntimeError("parent tiling seed mismatch")
    if any(int(row["num_tiles"]) <= 0 for row in tiling_rows):
        raise RuntimeError("parent tiling contains a nonpositive tile count")
    for seed in seeds:
        if sum(int(row["num_tiles"]) for row in tiling_rows if int(row["tiling_seed"]) == seed) != 20_736:
            raise RuntimeError(f"parent tiling does not contain exactly 20,736 tiles: {seed}")

    return {
        "run_manifest": run_manifest,
        "execution_manifest": execution,
        "resolved_config": resolved,
        "checkpoint_info": checkpoint,
        "heldout_panel": panel,
        "tiling_rows": tiling_rows,
        "tiling_seeds": seeds,
        "parent_hashes": parent_hashes,
        "parent_realpath": str(parent),
        "heldout_realpath": str(heldout_realpath),
        "checkpoint_realpath": str(checkpoint_realpath),
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "panel_models": sorted(panel_models),
    }


def validate_template_snapshot(
    parent: Path,
    audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    template_path = parent / "source_condition_templates.pt"
    actual_file_sha = gate.sha256_file(template_path)
    expected_file_sha = EXPECTED_PARENT_HASHES["source_condition_templates.pt"]
    if actual_file_sha != expected_file_sha or actual_file_sha != audit["parent_hashes"][template_path.name]:
        raise RuntimeError(
            f"template file changed after parent audit: actual={actual_file_sha} expected={expected_file_sha}"
        )
    templates = torch.load(template_path, map_location="cpu", weights_only=False)
    if set(templates) != {"global", "cell_mean", "cell_medoid"}:
        raise RuntimeError(f"unexpected template kinds: {set(templates)}")
    expected_cells = {(role, depth) for role in gate.ROLES for depth in range(12)}
    if set(templates["cell_mean"]) != expected_cells:
        raise RuntimeError("source cell-mean template grid is not exactly role x 12 depths")

    summary_rows = read_csv(parent / "condition_template_summary.csv")
    summary_index = {
        (str(row["role"]), int(row["canonical_depth"])): row
        for row in summary_rows
        if row["kind"] == "cell_mean"
    }
    if set(summary_index) != expected_cells:
        raise RuntimeError("parent cell-mean template summary grid is incomplete")
    cell_hash_items: list[str] = []
    for role, depth in sorted(expected_cells):
        value = templates["cell_mean"][(role, depth)]
        if set(value) != {"c_var", "c_patch"}:
            raise RuntimeError(f"unexpected template tensors at {role}/{depth}: {set(value)}")
        if tuple(value["c_var"].shape) != (256,) or tuple(value["c_patch"].shape) != (256,):
            raise RuntimeError(f"bad template geometry at {role}/{depth}: {value}")
        if not torch.isfinite(value["c_var"]).all() or not torch.isfinite(value["c_patch"]).all():
            raise RuntimeError(f"non-finite source template at {role}/{depth}")
        c_var_sha = gate.tensor_sha256(value["c_var"])
        c_patch_sha = gate.tensor_sha256(value["c_patch"])
        expected = summary_index[(role, depth)]
        if c_var_sha != expected["c_var_sha256"] or c_patch_sha != expected["c_patch_sha256"]:
            raise RuntimeError(f"template tensor hash mismatch at {role}/{depth}")
        cell_hash_items.append(f"{role}:{depth}:{c_var_sha}:{c_patch_sha}")

    distance_rows: list[dict[str, Any]] = []
    for mapping in wrong_condition_map():
        target = templates["cell_mean"][(mapping["target_role"], mapping["target_depth"])]["c_patch"].float()
        wrong = templates["cell_mean"][(mapping["template_role"], mapping["template_depth"])]["c_patch"].float()
        target_norm = float(target.norm())
        wrong_norm = float(wrong.norm())
        difference_norm = float((wrong - target).norm())
        cosine = float(torch.dot(target, wrong) / (target.norm() * wrong.norm()).clamp_min(1e-30))
        row = {
            **mapping,
            "tensor": "decoder_c_patch",
            "target_sha256": gate.tensor_sha256(target),
            "wrong_sha256": gate.tensor_sha256(wrong),
            "target_norm": target_norm,
            "wrong_norm": wrong_norm,
            "wrong_over_target_norm": wrong_norm / target_norm,
            "l2_distance": difference_norm,
            "relative_l2_distance": difference_norm / target_norm,
            "cosine": cosine,
        }
        if (
            row["target_sha256"] == row["wrong_sha256"]
            or not all(math.isfinite(float(row[key])) for key in (
                "target_norm",
                "wrong_norm",
                "wrong_over_target_norm",
                "l2_distance",
                "relative_l2_distance",
                "cosine",
            ))
            or target_norm <= 0.0
            or wrong_norm <= 0.0
            or difference_norm <= 0.0
        ):
            raise RuntimeError(f"degenerate wrong-template intervention: {row}")
        distance_rows.append(row)
    if len(distance_rows) != 144:
        raise RuntimeError(f"expected 144 separate wrong-template distances, got {len(distance_rows)}")
    snapshot = {
        "template_file_sha256": actual_file_sha,
        "condition_template_summary_sha256": gate.sha256_file(parent / "condition_template_summary.csv"),
        "cell_tensor_grid_sha256": gate.stable_hex(*cell_hash_items),
        "cell_count": len(expected_cells),
        "distance_rows": len(distance_rows),
    }
    return templates, distance_rows, snapshot


def write_preexecution_contract(
    *,
    args: argparse.Namespace,
    output: Path,
    audit: dict[str, Any],
) -> None:
    if not POSTRUN_ANALYZER.is_file():
        raise FileNotFoundError(f"frozen postrun analyzer is missing: {POSTRUN_ANALYZER}")
    dependency_hashes = audit_code_dependencies()
    resolved = {
        "parent_run_dir": audit["parent_realpath"],
        "heldout_source_realpath": audit["heldout_realpath"],
        "checkpoint_realpath": audit["checkpoint_realpath"],
        "output_dir": str(output),
        "device": str(args.device),
        "dtype": "parent checkpoint AMP policy",
        "batch_size": int(args.batch_size),
        "bootstrap_draws": int(args.bootstrap_draws),
        "bootstrap_seeds": {
            "paired_code": int(args.factorial_seed) + 99,
            "paired_c": int(args.factorial_seed) + 199,
        },
        "factorial_seed": int(args.factorial_seed),
        "persist_code_cache": bool(args.persist_code_cache),
        "execute": bool(args.execute),
        "cache_mode": "reuse exact parent panel/templates/A-B refs/tilings; new source-only z_dec cache",
        "target_access": False,
        "target_access_seal": TARGET_ACCESS_SEAL,
        "encoder_conditions": list(ENC_CONDITIONS),
        "code_conditions": list(CODE_CONDITIONS),
        "decoder_conditions": list(DEC_CONDITIONS),
        "formal_arms": len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(DEC_CONDITIONS),
        "unique_decoder_arms": len(COMPUTED_ARMS),
        "intervention_point": "z_dec=latent_norm(latent_slots.flatten(1)), shape [B,512]",
        "primary_metrics": (
            "raw full-matrix E_X/operator cosine plus paired block-bootstrap deltas and "
            "comparator/correct ratios; no heldout-fitted gain"
        ),
        "strong_weight_code_rule": (
            "PRIMARY fixed cell/cell: for within-row permutation and zero, both tilings, macro and micro, "
            "point comparator/correct raw E_X ratio >= 1.05 and paired one-sided ratio L95 > 1 (8 rows). "
            "Native/native is an analogous robustness tier; all 16 is a stronger context-robust screen. "
            "The depth+6 block donor is secondary"
        ),
        "directional_operator_structure_rule": (
            "before directional/operator-structure wording, under cell/cell correct-minus-within-row paired "
            "operator-cosine L95 must exceed zero for macro and micro on both tilings"
        ),
        "nonpass_rule": (
            "if ratio U95 <1.05, the >=5% effect is excluded at that endpoint; otherwise non-pass is "
            "inconclusive rather than falsification"
        ),
        "scope_limit": (
            "pass establishes causal W/tile-specific z contribution on the source panel only; it does not "
            "establish absolute reconstruction quality, transfer, or a global weight manifold"
        ),
        "correct_code_c_interaction_estimands": {
            "cells": {
                "CC": "C_enc=cell,C_dec=cell",
                "NC": "C_enc=native,C_dec=cell",
                "CN": "C_enc=cell,C_dec=native",
                "NN": "C_enc=native,C_dec=native",
            },
            "difference_in_differences": "NN - NC - CN + CC",
            "metrics": ["raw_E_X", "raw_E_W", "raw_operator_cosine", "raw_weight_cosine"],
            "aggregations": ["macro", "micro", *gate.ROLES],
            "status": "descriptive mechanism diagnostic; not a substitute for the primary paired code rule",
        },
        "postrun_analyzer": str(POSTRUN_ANALYZER),
        "postrun_analyzer_sha256": gate.sha256_file(POSTRUN_ANALYZER),
        "postrun_analyzer_command": (
            f"{sys.executable} {POSTRUN_ANALYZER} --run-dir {output} "
            f"--output-dir {output / 'postrun_analysis'}"
        ),
    }
    gate.write_json(output / "resolved_config.json", resolved)
    gate.write_json(output / "parent_artifact_hashes.json", audit["parent_hashes"])
    gate.write_json(output / "transitive_code_dependency_hashes.json", dependency_hashes)
    gate.write_json(output / "factorial_arms.json", arm_manifest())
    gate.write_json(output / "donor_block_map.json", donor_block_map())
    gate.write_json(output / "wrong_condition_map.json", wrong_condition_map())
    gate.write_json(
        output / "preexecution_contract.json",
        {
            "status": "LOCKED_BEFORE_WEIGHT_AE_OR_DISTRIBUTION_ENCODER_FORWARD",
            "parent_target_access": False,
            "target_data2vec_access": False,
            "parent_realpath": audit["parent_realpath"],
            "heldout_source_realpath": audit["heldout_realpath"],
            "checkpoint_realpath": audit["checkpoint_realpath"],
            "checkpoint_file_sha256": audit["checkpoint_file_sha256"],
            "expected_parent_hashes": EXPECTED_PARENT_HASHES,
            "parent_checkpoint": audit["checkpoint_info"],
            "parent_tiling_seeds": list(audit["tiling_seeds"]),
            "parent_panel_sha256": audit["parent_hashes"]["heldout_panel_manifest.json"],
            "parent_template_sha256": audit["parent_hashes"]["source_condition_templates.pt"],
            "implementation_sha256": gate.sha256_file(Path(__file__).resolve()),
            "resolved_config_sha256": gate.sha256_file(output / "resolved_config.json"),
            "arms_sha256": gate.sha256_file(output / "factorial_arms.json"),
            "donor_map_sha256": gate.sha256_file(output / "donor_block_map.json"),
            "wrong_condition_map_sha256": gate.sha256_file(output / "wrong_condition_map.json"),
            "postrun_analyzer_sha256": gate.sha256_file(POSTRUN_ANALYZER),
            "transitive_code_dependency_hashes_sha256": gate.sha256_file(
                output / "transitive_code_dependency_hashes.json"
            ),
            "target_access_seal_sha256": gate.sha256_file(output / "target_access_seal.json"),
        },
    )


def load_records(parent: Path, audit: dict[str, Any], seed: int, logger: logging.Logger) -> list[gate.MatrixRecord]:
    del parent  # The exact source-bank path is bound by audit_parent().
    heldout_root = Path(audit["heldout_realpath"])
    dataset = gate.build_dataset(heldout_root, seed)
    records: list[gate.MatrixRecord] = []
    for index, item in enumerate(audit["heldout_panel"]):
        context = ref_from_dict(item["context_a"])
        score = ref_from_dict(item["score_b"])
        if context.model_name != "vit_base_p16_224" or score.model_name != "vit_base_p16_224":
            raise RuntimeError(f"non-ViT-B heldout ref: {context}/{score}")
        if context.source_key != score.source_key or context.role != score.role or context.depth != score.depth:
            raise RuntimeError(f"context/score matrix identity mismatch: {context}/{score}")
        if context.weight_shape != gate.ROLE_SHAPES[context.role] or score.weight_shape != context.weight_shape:
            raise RuntimeError(f"heldout role geometry mismatch: {context}/{score}")
        W_a, X_a = gate.load_sample(dataset, context)
        W_b, X_b = gate.load_sample(dataset, score)
        if not torch.equal(W_a, W_b) or torch.equal(X_a, X_b):
            raise RuntimeError(f"heldout A/B contract failed: {context.source_key}")
        records.append(
            gate.MatrixRecord(
                model_name=context.model_name,
                depth=context.depth,
                canonical_depth=context.depth,
                role=context.role,
                layer_name=context.layer_name,
                source_key=context.source_key,
                context_ref=context,
                score_ref=score,
                W=W_a,
                X_context=X_a,
                X_score=X_b,
            )
        )
        logger.info(
            "stage=heldout_load matrix=%s/72 depth=%s role=%s shape=%s",
            index + 1,
            context.depth,
            context.role,
            tuple(W_a.shape),
        )
    expected = {(depth, role) for depth in range(12) for role in gate.ROLES}
    if {(record.depth, record.role) for record in records} != expected:
        raise RuntimeError("heldout panel is not an exact 12x6 depth-role grid")
    return records


def validate_locked_tilings(
    records: list[gate.MatrixRecord],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    parent_index = csv_sha_index(audit["tiling_rows"])
    rows: list[dict[str, Any]] = []
    for seed in audit["tiling_seeds"]:
        tile_total = 0
        for record in records:
            row = gate.validate_tiling(record, gate.make_tiling(record, seed))
            expected_sha = parent_index[(record.role, record.canonical_depth, seed)]
            if row["partition_sha256"] != expected_sha:
                raise RuntimeError(
                    f"locked tiling mismatch seed={seed} depth={record.depth} role={record.role}"
                )
            rows.append(row)
            tile_total += int(row["num_tiles"])
        if tile_total != 20_736:
            raise RuntimeError(f"locked tile total mismatch: seed={seed} total={tile_total}")
    return rows


def build_code_derangement_manifest(
    records: list[gate.MatrixRecord],
    tiling_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_cell = {(record.depth, record.role): record for record in records}
    if len(by_cell) != 72:
        raise RuntimeError("derangement requires an exact 12x6 record grid")
    rows: list[dict[str, Any]] = []
    for role in gate.ROLES:
        donor_depths = [(depth + 6) % 12 for depth in range(12)]
        if sorted(donor_depths) != list(range(12)) or any(a == b for a, b in enumerate(donor_depths)):
            raise RuntimeError(f"donor depth map is not a no-fixed-point bijection: {role}/{donor_depths}")
    for seed in tiling_seeds:
        for target in records:
            donor = by_cell[((target.depth + 6) % 12, target.role)]
            if (
                donor.depth == target.depth
                or donor.source_key == target.source_key
                or donor.role != target.role
                or tuple(donor.W.shape) != tuple(target.W.shape)
            ):
                raise RuntimeError(f"invalid donor block: target={target.source_key} donor={donor.source_key}")
            tiling = gate.make_tiling(target, seed)
            column_count = len(tiling.cols)
            column_permutation = within_row_column_permutation(tiling)
            ordinal = 0
            for row_idx, row in enumerate(tiling.rows):
                for col_idx, col in enumerate(tiling.cols):
                    within_col_idx = int(column_permutation[col_idx])
                    within_ordinal = row_idx * column_count + within_col_idx
                    if within_ordinal == ordinal:
                        raise RuntimeError(f"within-row code permutation has a fixed point: {target.source_key}")
                    common = {
                            "tiling_seed": seed,
                            "role": target.role,
                            "target_depth": target.depth,
                            "target_source_key": target.source_key,
                            "target_tile_ordinal": ordinal,
                            "target_row_group_ordinal": row_idx,
                            "target_col_group_ordinal": col_idx,
                            "target_row_indices_sha256": gate.tensor_sha256(row),
                            "target_col_indices_sha256": gate.tensor_sha256(col),
                            "role_preserved": True,
                            "shape_preserved": True,
                    }
                    rows.append(
                        {
                            **common,
                            "code_condition": "deranged_block",
                            "donor_depth": donor.depth,
                            "donor_source_key": donor.source_key,
                            "donor_tile_ordinal": ordinal,
                            "donor_row_group_ordinal": row_idx,
                            "donor_col_group_ordinal": col_idx,
                            "donor_row_indices_sha256": gate.tensor_sha256(row),
                            "donor_col_indices_sha256": gate.tensor_sha256(col),
                            "target_partition_applied_to_donor_W": True,
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "code_condition": "permuted_within_row",
                            "donor_depth": target.depth,
                            "donor_source_key": target.source_key,
                            "donor_tile_ordinal": within_ordinal,
                            "donor_row_group_ordinal": row_idx,
                            "donor_col_group_ordinal": within_col_idx,
                            "donor_row_indices_sha256": gate.tensor_sha256(row),
                            "donor_col_indices_sha256": gate.tensor_sha256(tiling.cols[within_col_idx]),
                            "target_partition_applied_to_donor_W": False,
                        }
                    )
                    ordinal += 1
            if ordinal != tiling.num_tiles:
                raise RuntimeError(f"derangement tile enumeration mismatch: {target.source_key}")
    expected_rows = len(tiling_seeds) * 20_736 * 2
    keys = {
        (
            int(row["tiling_seed"]),
            str(row["code_condition"]),
            str(row["target_source_key"]),
            int(row["target_tile_ordinal"]),
        )
        for row in rows
    }
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise RuntimeError(f"derangement manifest duplicate/missing rows: {len(rows)}/{len(keys)}")
    return rows


def within_row_column_permutation(tiling: gate.Tiling) -> torch.Tensor:
    column_count = len(tiling.cols)
    if column_count <= 1 or column_count % 2:
        raise RuntimeError(f"within-row permutation requires an even column count >1: {column_count}")
    permutation = (torch.arange(column_count, dtype=torch.long) + column_count // 2) % column_count
    if sorted(int(value) for value in permutation) != list(range(column_count)):
        raise RuntimeError(f"within-row column map is not bijective: {permutation.tolist()}")
    if bool(torch.any(permutation == torch.arange(column_count))):
        raise RuntimeError(f"within-row column map has a fixed point: {permutation.tolist()}")
    return permutation


def validate_code_derangement_manifest(
    rows: list[dict[str, Any]],
    tiling_seeds: tuple[int, ...],
) -> dict[str, Any]:
    expected_per_condition = len(tiling_seeds) * 20_736
    counts = {
        condition: sum(row["code_condition"] == condition for row in rows)
        for condition in ("permuted_within_row", "deranged_block")
    }
    if set(counts.values()) != {expected_per_condition}:
        raise RuntimeError(f"derangement condition census mismatch: {counts}/{expected_per_condition}")
    within = [row for row in rows if row["code_condition"] == "permuted_within_row"]
    if not all(
        int(row["donor_depth"]) == int(row["target_depth"])
        and row["donor_source_key"] == row["target_source_key"]
        and int(row["donor_row_group_ordinal"]) == int(row["target_row_group_ordinal"])
        and int(row["donor_col_group_ordinal"]) != int(row["target_col_group_ordinal"])
        and int(row["donor_tile_ordinal"]) != int(row["target_tile_ordinal"])
        and row["donor_row_indices_sha256"] == row["target_row_indices_sha256"]
        and row["donor_col_indices_sha256"] != row["target_col_indices_sha256"]
        and row["target_partition_applied_to_donor_W"] is False
        for row in within
    ):
        raise RuntimeError("within-row manifest violates the same-matrix/no-fixed-point contract")
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in within:
        grouped[
            (
                int(row["tiling_seed"]),
                str(row["target_source_key"]),
                int(row["target_row_group_ordinal"]),
            )
        ].append(row)
    if not all(
        sorted(int(row["target_col_group_ordinal"]) for row in group)
        == sorted(int(row["donor_col_group_ordinal"]) for row in group)
        for group in grouped.values()
    ):
        raise RuntimeError("within-row manifest does not preserve the column-code multiset in every row group")
    block = [row for row in rows if row["code_condition"] == "deranged_block"]
    if not all(
        int(row["donor_depth"]) == (int(row["target_depth"]) + 6) % 12
        and row["donor_source_key"] != row["target_source_key"]
        and int(row["donor_tile_ordinal"]) == int(row["target_tile_ordinal"])
        and row["donor_row_indices_sha256"] == row["target_row_indices_sha256"]
        and row["donor_col_indices_sha256"] == row["target_col_indices_sha256"]
        and row["target_partition_applied_to_donor_W"] is True
        for row in block
    ):
        raise RuntimeError("depth+6 manifest violates the same-role target-partition contract")
    return {
        "pass": True,
        "rows": len(rows),
        "counts": counts,
        "primary_control": "same_matrix_same_row_group_no_fixed_point_column_code_permutation",
        "within_row_groups": len(grouped),
        "within_row_bijective_per_group": True,
        "depth_plus_6_secondary": True,
    }


def build_model(
    *,
    checkpoint_path: Path,
    device: torch.device,
    no_amp: bool,
) -> tuple[torch.nn.Module, bool, torch.dtype | None, dict[str, Any]]:
    checkpoint_sha256 = gate.sha256_file(checkpoint_path)
    if checkpoint_sha256 != gate.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint changed before model construction: {checkpoint_sha256} != {gate.EXPECTED_CHECKPOINT_SHA256}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["config"])
    model_cfg = gate._build_model_cfg(cfg)
    model_cfg.big_vae.use_latent_sampling = False
    model_cfg.big_vae.rope_2d_coord_kind = "raw"
    model = gate.build_weight_quantile_vae(model_cfg).to(device)
    model.load_state_dict(gate._normalize_model_state_dict_keys(payload["model_state"]), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    amp_enabled, amp_dtype = gate._resolve_amp(cfg, device)
    amp_enabled = bool(amp_enabled and not no_amp)

    decode_signature = inspect.signature(model._decode_from_latent_slots)
    signature_defaults = {
        name: decode_signature.parameters[name].default
        for name in (
            "debug_decoder_kv_source",
            "debug_query_hint",
            "debug_direct_from_encoder_tokens",
        )
    }
    contract = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(payload.get("step", 0)),
        "strict_load": True,
        "rope_2d_coord_kind": str(model.rope_2d_coord_kind),
        "use_latent_sampling": bool(model.cfg.big_vae.use_latent_sampling),
        "use_encoder_mu_head": bool(model.cfg.big_vae.use_encoder_mu_head),
        "disable_z_shortcut": bool(model.cfg.big_vae.disable_z_shortcut),
        "patch_tokenizer_kind": str(model.patch_tokenizer_kind),
        "distribution_encoder_conditioning_kind": str(model.distribution_encoder_conditioning_kind),
        "num_latents": int(model.cfg.big_vae.num_latents),
        "d_lat": int(model.cfg.big_vae.d_lat),
        "flat_lat_dim": int(model.flat_lat_dim),
        "patch_size": int(model.cfg.patch_size),
        "locked_tile_shape": [64, 64],
        "locked_patch_geometry": {"T": 4, "d_in_pad": 64, "all_patch_mask_valid": True},
        "decoder_defaults": signature_defaults,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
    }
    expected = {
        "checkpoint_sha256": gate.EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_step": gate.EXPECTED_CHECKPOINT_STEP,
        "rope_2d_coord_kind": "raw",
        "use_latent_sampling": False,
        "use_encoder_mu_head": False,
        "disable_z_shortcut": True,
        "patch_tokenizer_kind": "conditioned_mlp",
        "distribution_encoder_conditioning_kind": "token_adapter",
        "num_latents": 8,
        "d_lat": 64,
        "flat_lat_dim": 512,
        "patch_size": 16,
        "decoder_defaults": {
            "debug_decoder_kv_source": "latents",
            "debug_query_hint": "none",
            "debug_direct_from_encoder_tokens": False,
        },
    }
    for key, expected_value in expected.items():
        if contract[key] != expected_value:
            raise RuntimeError(f"canonical decoder-path contract failed at {key}: {contract}")
    return model, amp_enabled, amp_dtype, contract


def masks(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    d_in_mask = torch.ones((batch, 64), dtype=torch.bool, device=device)
    d_out_mask = torch.ones((batch, 64), dtype=torch.bool, device=device)
    # Every locked evaluation tile is exactly 64x64 and the canonical patch
    # size is 16, hence all four input patches are valid.
    T, d_in_pad = 4, 64
    patch_mask = torch.ones((batch, T), dtype=torch.bool, device=device)
    return d_in_mask, d_out_mask, patch_mask, T, d_in_pad


def fixed_condition(
    template: dict[str, torch.Tensor],
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c_var = template["c_var"].to(device=device, dtype=dtype).view(1, 1, 1, -1).expand(
        batch, 4, 16, -1
    )
    c_patch = template["c_patch"].to(device=device, dtype=dtype).view(1, 1, -1).expand(
        batch, 4, -1
    )
    return c_var, c_patch, c_var.mean(dim=2)


@torch.inference_mode()
def native_condition(
    model: torch.nn.Module,
    X_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(X_batch.shape[0])
    device = X_batch.device
    x_mask = torch.ones((batch, int(X_batch.shape[1])), dtype=torch.bool, device=device)
    d_in_mask = torch.ones((batch, 64), dtype=torch.bool, device=device)
    T, d_in_pad, patch_mask, structural, c_var, c_patch, c_pooled = model._encode_distribution_context(
        X_batch,
        x_mask=x_mask,
        d_in_mask=d_in_mask,
    )
    if c_var is None or c_patch is None or c_pooled is None:
        raise RuntimeError("native distribution encoder returned None")
    if (
        int(model.cfg.patch_size) != 16
        or int(T) != 4
        or int(d_in_pad) != 64
        or tuple(patch_mask.shape) != (batch, 4)
        or not bool(torch.all(patch_mask))
        or tuple(structural.shape) != (batch, 4)
        or not bool(torch.all(structural))
        or tuple(c_var.shape[:3]) != (batch, 4, 16)
        or tuple(c_patch.shape[:2]) != (batch, 4)
        or tuple(c_pooled.shape[:2]) != (batch, 4)
    ):
        raise RuntimeError(
            "native patch geometry mismatch: "
            f"patch_size={model.cfg.patch_size} T={T} d_in_pad={d_in_pad} "
            f"patch_mask={tuple(patch_mask.shape)}/{bool(torch.all(patch_mask))} "
            f"structural={tuple(structural.shape)}/{bool(torch.all(structural))} "
            f"C={tuple(c_var.shape)}/{tuple(c_patch.shape)}/{tuple(c_pooled.shape)}"
        )
    return c_var, c_patch, c_pooled


@torch.inference_mode()
def encode_z_dec(
    model: torch.nn.Module,
    W: torch.Tensor,
    condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    return_encoder_tokens: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c_var, c_patch, c_pooled = condition
    batch = int(W.shape[0])
    d_in_mask, d_out_mask, patch_mask, T, d_in_pad = masks(batch, W.device)
    latents, debug = model._encode_latent_slots(
        W,
        T=T,
        d_in_pad=d_in_pad,
        patch_mask=patch_mask,
        d_out_mask=d_out_mask,
        dist_var_by_patch=c_var,
        dist_patch_by_patch=c_patch,
        dist_var_pooled=c_pooled,
        return_debug_info=True,
    )
    z_dec = model.latent_norm(latents.reshape(batch, model.flat_lat_dim))
    if return_encoder_tokens:
        return z_dec, latents, debug["encoder_patch_tokens"]
    return z_dec


@torch.inference_mode()
def decode_z_dec(
    model: torch.nn.Module,
    z_dec: torch.Tensor,
    c_patch: torch.Tensor,
) -> torch.Tensor:
    batch = int(z_dec.shape[0])
    d_in_mask, d_out_mask, patch_mask, T, d_in_pad = masks(batch, z_dec.device)
    W_hat, _mu, _logvar, _dirs = model._decode_from_decoder_latent(
        z_dec,
        dist_patch_by_patch=c_patch,
        patch_mask=patch_mask,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        d_in=64,
        d_out=64,
        d_in_pad=d_in_pad,
        T=T,
    )
    return W_hat


def tile_rows(tiling: gate.Tiling) -> tuple[torch.Tensor, ...]:
    return tuple(row for row in tiling.rows for _col in tiling.cols)


@torch.inference_mode()
def run_path_preflight(
    *,
    model: torch.nn.Module,
    records: list[gate.MatrixRecord],
    templates: dict[str, Any],
    tiling_seed: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    probes = [
        next(record for record in records if record.depth == 0 and record.role == "attn_query"),
        next(record for record in records if record.depth == 0 and record.role == "ffn_up"),
    ]
    rows: list[dict[str, Any]] = []
    for record in probes:
        tiling = gate.make_tiling(record, tiling_seed)
        all_tiles = gate.split_tiles(record.W, tiling)
        all_rows = tile_rows(tiling)
        for execution_batch in (1, 2):
            W = all_tiles[:execution_batch].to(device)
            X = torch.stack(
                [record.X_context[:, row] for row in all_rows[:execution_batch]]
            ).to(device)
            with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                native = native_condition(model, X)
                cell = fixed_condition(
                    templates["cell_mean"][(record.role, record.canonical_depth)],
                    batch=execution_batch,
                    device=device,
                    dtype=W.dtype,
                )
                for enc_name, condition in (("cell", cell), ("native", native)):
                    z_dec, latents, encoder_tokens = encode_z_dec(
                        model,
                        W,
                        condition,
                        return_encoder_tokens=True,
                    )
                    c_patch = condition[1]
                    d_in_mask, d_out_mask, patch_mask, T, d_in_pad = masks(execution_batch, device)
                    common = {
                        "dist_patch_by_patch": c_patch,
                        "patch_mask": patch_mask,
                        "d_in_mask": d_in_mask,
                        "d_out_mask": d_out_mask,
                        "d_in": 64,
                        "d_out": 64,
                        "d_in_pad": d_in_pad,
                        "T": T,
                    }
                    slots_out = model._decode_from_latent_slots(
                        latents,
                        **common,
                        encoder_patch_tokens=encoder_tokens,
                    )[0]
                    z_out = decode_z_dec(model, z_dec, c_patch)
                    none_token_out = model._decode_from_latent_slots(
                        latents,
                        **common,
                        encoder_patch_tokens=None,
                    )[0]
                    zero_token_out = model._decode_from_latent_slots(
                        latents,
                        **common,
                        encoder_patch_tokens=torch.zeros_like(encoder_tokens),
                    )[0]
                    shortcut_forced_off = model._decode_from_decoder_latent(
                        z_dec,
                        **common,
                        disable_z_shortcut=True,
                    )[0]
                    rows.append(
                        {
                            "depth": record.depth,
                            "role": record.role,
                            "execution_batch": execution_batch,
                            "encoder_condition": enc_name,
                            "slots_vs_explicit_z_bit_exact": bool(torch.equal(slots_out, z_out)),
                            "slots_vs_explicit_z_max_abs": float(
                                (slots_out.float() - z_out.float()).abs().max()
                            ),
                            "encoder_tokens_actual_vs_none_bit_exact": bool(
                                torch.equal(slots_out, none_token_out)
                            ),
                            "encoder_tokens_actual_vs_zero_bit_exact": bool(
                                torch.equal(slots_out, zero_token_out)
                            ),
                            "z_shortcut_default_vs_forced_off_bit_exact": bool(
                                torch.equal(z_out, shortcut_forced_off)
                            ),
                            "z_dec_shape": list(z_dec.shape),
                            "z_dec_finite": bool(torch.isfinite(z_dec).all()),
                            "z_dec_nonzero": bool(torch.count_nonzero(z_dec) > 0),
                            "patch_size": int(model.cfg.patch_size),
                            "T": T,
                            "d_in_pad": d_in_pad,
                            "all_patch_mask_valid": bool(torch.all(patch_mask)),
                        }
                    )
    required = all(
        bool(row[key])
        for row in rows
        for key in (
            "slots_vs_explicit_z_bit_exact",
            "encoder_tokens_actual_vs_none_bit_exact",
            "encoder_tokens_actual_vs_zero_bit_exact",
            "z_shortcut_default_vs_forced_off_bit_exact",
            "z_dec_finite",
            "z_dec_nonzero",
        )
    ) and all(
        float(row["slots_vs_explicit_z_max_abs"]) <= 1e-6
        and int(row["patch_size"]) == 16
        and int(row["T"]) == 4
        and int(row["d_in_pad"]) == 64
        and bool(row["all_patch_mask_valid"])
        for row in rows
    )
    return {"pass": required, "probes": rows}


def code_key(seed: int, record: gate.MatrixRecord, enc: str, code: str) -> str:
    return f"seed={seed}|depth={record.depth}|role={record.role}|enc={enc}|code={code}"


def patch_key(seed: int, record: gate.MatrixRecord) -> str:
    return f"seed={seed}|depth={record.depth}|role={record.role}|native_c_patch"


def code_summary_row(
    *,
    seed: int,
    record: gate.MatrixRecord,
    enc: str,
    code: str,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    value = tensor.float()
    return {
        "tiling_seed": seed,
        "depth": record.depth,
        "role": record.role,
        "encoder_condition": enc,
        "code_condition": code,
        "tiles": int(tensor.shape[0]),
        "code_dim": int(tensor.shape[1]),
        "stored_dtype": str(tensor.dtype),
        "tensor_sha256": gate.tensor_sha256(tensor),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "mean_l2": float(value.norm(dim=1).mean()),
        "min_l2": float(value.norm(dim=1).min()),
        "max_l2": float(value.norm(dim=1).max()),
        "finite": bool(torch.isfinite(value).all()),
    }


@torch.inference_mode()
def build_seed_cache(
    *,
    seed: int,
    records: list[gate.MatrixRecord],
    templates: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_cell = {(record.depth, record.role): record for record in records}
    codes: dict[str, torch.Tensor] = {}
    native_patches: dict[str, torch.Tensor] = {}
    summary: list[dict[str, Any]] = []
    pair_summary: list[dict[str, Any]] = []
    pair_tiles: list[dict[str, Any]] = []
    native_row_diagnostics: list[dict[str, Any]] = []
    started = time.monotonic()
    for record_idx, record in enumerate(records):
        donor = by_cell[((record.depth + 6) % 12, record.role)]
        tiling = gate.make_tiling(record, seed)
        target_tiles = gate.split_tiles(record.W, tiling)
        # Causal block intervention: donor W uses the target partition, so global
        # coordinates and target C_enc remain fixed while block weights change.
        donor_tiles = gate.split_tiles(donor.W, tiling)
        rows_for_tiles = tile_rows(tiling)
        code_parts: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
        native_var_parts: list[torch.Tensor] = []
        native_patch_parts: list[torch.Tensor] = []
        native_pooled_parts: list[torch.Tensor] = []
        for begin in range(0, int(target_tiles.shape[0]), batch_size):
            end = min(begin + batch_size, int(target_tiles.shape[0]))
            W_target = target_tiles[begin:end].to(device)
            W_donor = donor_tiles[begin:end].to(device)
            X_native = torch.stack(
                [record.X_context[:, row] for row in rows_for_tiles[begin:end]]
            ).to(device)
            with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                native = native_condition(model, X_native)
                cell = fixed_condition(
                    templates["cell_mean"][(record.role, record.canonical_depth)],
                    batch=end - begin,
                    device=device,
                    dtype=W_target.dtype,
                )
                native_var_parts.append(native[0].detach().cpu().contiguous())
                native_patch_parts.append(native[1].detach().cpu().contiguous())
                native_pooled_parts.append(native[2].detach().cpu().contiguous())
                for enc_name, condition in (("cell", cell), ("native", native)):
                    target_z = encode_z_dec(model, W_target, condition)
                    donor_z = encode_z_dec(model, W_donor, condition)
                    assert isinstance(target_z, torch.Tensor) and isinstance(donor_z, torch.Tensor)
                    code_parts[(enc_name, "correct")].append(target_z.detach().cpu().contiguous())
                    code_parts[(enc_name, "deranged_block")].append(donor_z.detach().cpu().contiguous())
            logger.info(
                "stage=code_encode seed=%s matrix=%s/72 depth=%s role=%s donor_depth=%s tiles=%s/%s elapsed=%.1fs",
                seed,
                record_idx + 1,
                record.depth,
                record.role,
                donor.depth,
                end,
                int(target_tiles.shape[0]),
                time.monotonic() - started,
            )
        native_var = torch.cat(native_var_parts)
        native_patch = torch.cat(native_patch_parts)
        native_pooled = torch.cat(native_pooled_parts)
        row_count = len(tiling.rows)
        column_count = len(tiling.cols)
        condition_diagnostic: dict[str, Any] = {
            "tiling_seed": seed,
            "depth": record.depth,
            "role": record.role,
            "source_key": record.source_key,
            "row_groups": row_count,
            "columns_per_row_group": column_count,
            "x_row_templates_sha256": gate.tensor_sha256(
                torch.stack([record.X_context[:, row] for row in tiling.rows]).contiguous()
            ),
            "native_context_basis": (
                "X_context_A[:,row_indices] is identical for every column tile in a row group; "
                "the deterministic distribution encoder must return bit-identical C"
            ),
        }
        for condition_name, condition_tensor in {
            "c_var": native_var,
            "c_patch": native_patch,
            "c_pooled": native_pooled,
        }.items():
            grouped = condition_tensor.view(row_count, column_count, *condition_tensor.shape[1:])
            reference = grouped[:, :1].expand_as(grouped)
            max_abs = float((grouped.float() - reference.float()).abs().max())
            exact = bool(torch.equal(grouped, reference))
            condition_diagnostic[f"{condition_name}_identical_within_row_group"] = exact
            condition_diagnostic[f"{condition_name}_max_abs_within_row_group"] = max_abs
            condition_diagnostic[f"{condition_name}_row_templates_sha256"] = gate.tensor_sha256(
                grouped[:, 0].contiguous()
            )
            if not exact:
                raise RuntimeError(
                    f"native {condition_name} changed across column tiles with identical row X: "
                    f"seed={seed} depth={record.depth} role={record.role} max_abs={max_abs}"
                )
        native_row_diagnostics.append(condition_diagnostic)
        native_patches[patch_key(seed, record)] = native_patch
        summary.append(
            {
                "tiling_seed": seed,
                "depth": record.depth,
                "role": record.role,
                "encoder_condition": "native_c_patch",
                "code_condition": "not_applicable",
                "tiles": int(native_patch.shape[0]),
                "code_dim": int(native_patch.shape[-1]),
                "stored_dtype": str(native_patch.dtype),
                "tensor_sha256": gate.tensor_sha256(native_patch),
                "mean": float(native_patch.float().mean()),
                "std": float(native_patch.float().std(unbiased=False)),
                "mean_l2": float(native_patch.float().flatten(1).norm(dim=1).mean()),
                "min_l2": float(native_patch.float().flatten(1).norm(dim=1).min()),
                "max_l2": float(native_patch.float().flatten(1).norm(dim=1).max()),
                "finite": bool(torch.isfinite(native_patch).all()),
            }
        )
        local_codes: dict[tuple[str, str], torch.Tensor] = {}
        for (enc_name, code_name), parts in sorted(code_parts.items()):
            value = torch.cat(parts)
            if tuple(value.shape) != (tiling.num_tiles, 512):
                raise RuntimeError(f"bad code shape: {record.source_key}/{enc_name}/{code_name}/{value.shape}")
            key = code_key(seed, record, enc_name, code_name)
            codes[key] = value
            local_codes[(enc_name, code_name)] = value
            summary.append(
                code_summary_row(
                    seed=seed,
                    record=record,
                    enc=enc_name,
                    code=code_name,
                    tensor=value,
                )
            )
        column_permutation = within_row_column_permutation(tiling)
        for enc_name in ENC_CONDITIONS:
            correct = local_codes[(enc_name, "correct")]
            within_row = (
                correct.view(row_count, column_count, 512)[:, column_permutation, :]
                .reshape(tiling.num_tiles, 512)
                .contiguous()
            )
            local_codes[(enc_name, "permuted_within_row")] = within_row
            codes[code_key(seed, record, enc_name, "permuted_within_row")] = within_row
            summary.append(
                code_summary_row(
                    seed=seed,
                    record=record,
                    enc=enc_name,
                    code="permuted_within_row",
                    tensor=within_row,
                )
            )
        for enc_name in ENC_CONDITIONS:
            correct = local_codes[(enc_name, "correct")].float()
            for comparator in ("permuted_within_row", "deranged_block"):
                donor_code = local_codes[(enc_name, comparator)].float()
                correct_sha = gate.tensor_sha256(correct)
                donor_sha = gate.tensor_sha256(donor_code)
                correct_norm = correct.norm(dim=1)
                donor_norm = donor_code.norm(dim=1)
                difference_norm = (correct - donor_code).norm(dim=1)
                cosine = (correct * donor_code).sum(dim=1) / (correct_norm * donor_norm).clamp_min(1e-30)
                relative_distance = difference_norm / correct_norm.clamp_min(1e-30)
                if (
                    correct_sha == donor_sha
                    or not torch.isfinite(correct).all()
                    or not torch.isfinite(donor_code).all()
                    or not torch.isfinite(difference_norm).all()
                    or not torch.isfinite(cosine).all()
                    or not torch.isfinite(relative_distance).all()
                    or not bool(torch.all(correct_norm > 0))
                    or not bool(torch.all(donor_norm > 0))
                    or not bool(torch.all(difference_norm > 0))
                ):
                    raise RuntimeError(
                        f"correct/donor latent separation failed: seed={seed} depth={record.depth} "
                        f"role={record.role} enc={enc_name} comparator={comparator} "
                        f"hashes={correct_sha}/{donor_sha}"
                    )
                pair_summary.append(
                    {
                        "tiling_seed": seed,
                        "target_depth": record.depth,
                        "target_role": record.role,
                        "target_source_key": record.source_key,
                        "donor_depth": donor.depth if comparator == "deranged_block" else record.depth,
                        "donor_source_key": donor.source_key if comparator == "deranged_block" else record.source_key,
                        "encoder_condition": enc_name,
                        "comparator": comparator,
                        "tiles": int(correct.shape[0]),
                        "correct_sha256": correct_sha,
                        "donor_sha256": donor_sha,
                        "hashes_unequal": True,
                        "all_tile_differences_nonzero": True,
                        "correct_mean_norm": float(correct_norm.mean()),
                        "donor_mean_norm": float(donor_norm.mean()),
                        "mean_difference_norm": float(difference_norm.mean()),
                        "min_difference_norm": float(difference_norm.min()),
                        "max_difference_norm": float(difference_norm.max()),
                        "mean_relative_distance": float(relative_distance.mean()),
                        "mean_cosine": float(cosine.mean()),
                        "min_cosine": float(cosine.min()),
                        "max_cosine": float(cosine.max()),
                        "finite": True,
                    }
                )
                for tile_ordinal in range(int(correct.shape[0])):
                    pair_tiles.append(
                        {
                            "tiling_seed": seed,
                            "target_depth": record.depth,
                            "target_role": record.role,
                            "target_source_key": record.source_key,
                            "donor_depth": donor.depth if comparator == "deranged_block" else record.depth,
                            "donor_source_key": donor.source_key if comparator == "deranged_block" else record.source_key,
                            "tile_ordinal": tile_ordinal,
                            "encoder_condition": enc_name,
                            "comparator": comparator,
                            "correct_norm": float(correct_norm[tile_ordinal]),
                            "donor_norm": float(donor_norm[tile_ordinal]),
                            "difference_norm": float(difference_norm[tile_ordinal]),
                            "relative_distance": float(relative_distance[tile_ordinal]),
                            "cosine": float(cosine[tile_ordinal]),
                        }
                    )
    expected_code_tensors = 72 * len(ENC_CONDITIONS) * len(CODE_WITH_ENCODER)
    if len(codes) != expected_code_tensors or len(native_patches) != 72:
        raise RuntimeError(f"incomplete seed cache: codes={len(codes)} patches={len(native_patches)}")
    if not all(bool(row["finite"]) for row in summary):
        raise RuntimeError("non-finite code/context cache")
    expected_pair_comparators = 2
    if (
        len(pair_summary) != 72 * len(ENC_CONDITIONS) * expected_pair_comparators
        or len(pair_tiles) != 20_736 * len(ENC_CONDITIONS) * expected_pair_comparators
    ):
        raise RuntimeError(
            f"latent pair diagnostics incomplete: summary={len(pair_summary)} tiles={len(pair_tiles)}"
        )
    if len(native_row_diagnostics) != 72:
        raise RuntimeError(f"native row-condition diagnostics incomplete: {len(native_row_diagnostics)}")
    return codes, native_patches, summary, pair_summary, pair_tiles, native_row_diagnostics


def decoder_patch(
    *,
    dec_condition: str,
    record: gate.MatrixRecord,
    begin: int,
    end: int,
    seed: int,
    templates: dict[str, Any],
    native_patches: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch = end - begin
    if dec_condition == "native":
        return native_patches[patch_key(seed, record)][begin:end].to(device)
    if dec_condition == "cell":
        template = templates["cell_mean"][(record.role, record.canonical_depth)]
    elif dec_condition == "wrong_role_same_depth":
        wrong_role = gate.ROLES[(gate.ROLES.index(record.role) + 1) % len(gate.ROLES)]
        template = templates["cell_mean"][(wrong_role, record.canonical_depth)]
    elif dec_condition == "same_role_wrong_depth":
        wrong_depth = (record.canonical_depth + 6) % 12
        template = templates["cell_mean"][(record.role, wrong_depth)]
    else:
        raise ValueError(dec_condition)
    return template["c_patch"].to(device=device, dtype=dtype).view(1, 1, -1).expand(batch, 4, -1)


def metric_row(
    *,
    record: gate.MatrixRecord,
    seed: int,
    arm: Arm,
    prediction: torch.Tensor,
    num_tiles: int,
) -> dict[str, Any]:
    stats = gate.sufficient_stats(record.W, prediction, record.X_score)
    raw_w_cosine = stats["w_dot"] / math.sqrt(max(stats["w_target"] * stats["w_pred"], 1e-300))
    raw_x_cosine = stats["x_dot"] / math.sqrt(max(stats["x_target"] * stats["x_pred"], 1e-300))
    return {
        "tiling_seed": seed,
        "encoder_condition": arm.encoder_condition,
        "code_condition": arm.code_condition,
        "decoder_condition": arm.decoder_condition,
        "computed_arm": arm.key,
        "depth": record.depth,
        "canonical_depth": record.canonical_depth,
        "role": record.role,
        "layer_name": record.layer_name,
        "source_key": record.source_key,
        "shape": "x".join(str(int(value)) for value in record.W.shape),
        "num_tiles": num_tiles,
        "donor_depth": (record.depth + 6) % 12 if arm.code_condition == "deranged_block" else -1,
        "decoder_template_kind": "heldout_context_a" if arm.decoder_condition == "native" else "source_cell_mean",
        "decoder_template_role": (
            ""
            if arm.decoder_condition == "native"
            else (
                gate.ROLES[(gate.ROLES.index(record.role) + 1) % len(gate.ROLES)]
                if arm.decoder_condition == "wrong_role_same_depth"
                else record.role
            )
        ),
        "decoder_template_depth": (
            -1
            if arm.decoder_condition == "native"
            else (
                (record.canonical_depth + 6) % 12
                if arm.decoder_condition == "same_role_wrong_depth"
                else record.canonical_depth
            )
        ),
        **stats,
        "raw_E_W": stats["w_error"] / stats["w_target"],
        "raw_E_X": stats["x_error"] / stats["x_target"],
        "raw_weight_cosine": raw_w_cosine,
        "raw_operator_cosine": raw_x_cosine,
        "raw_weight_norm_ratio": math.sqrt(stats["w_pred"] / stats["w_target"]),
        "raw_operator_norm_ratio": math.sqrt(stats["x_pred"] / stats["x_target"]),
        "prediction_sha256": gate.tensor_sha256(prediction),
        "zero_code_reused_across_encoder_labels": arm.code_condition == "zero",
    }


@torch.inference_mode()
def decode_seed_factorial(
    *,
    seed: int,
    records: list[gate.MatrixRecord],
    templates: dict[str, Any],
    codes: dict[str, torch.Tensor],
    native_patches: dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for record_idx, record in enumerate(records):
        tiling = gate.make_tiling(record, seed)
        predictions: dict[str, list[torch.Tensor]] = {arm.key: [] for arm in COMPUTED_ARMS}
        tile_count = tiling.num_tiles
        for begin in range(0, tile_count, batch_size):
            end = min(begin + batch_size, tile_count)
            batch = end - begin
            for arm in COMPUTED_ARMS:
                if arm.code_condition == "zero":
                    # This is exact decoder-entry zero.  Zeroing raw slots would
                    # retain latent_norm's learned bias and is not the intended intervention.
                    z_dec = torch.zeros((batch, 512), device=device, dtype=torch.float32)
                else:
                    z_dec = codes[code_key(seed, record, arm.encoder_condition, arm.code_condition)][
                        begin:end
                    ].to(device)
                c_patch = decoder_patch(
                    dec_condition=arm.decoder_condition,
                    record=record,
                    begin=begin,
                    end=end,
                    seed=seed,
                    templates=templates,
                    native_patches=native_patches,
                    device=device,
                    dtype=torch.float32,
                )
                with gate._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    prediction = decode_z_dec(model, z_dec, c_patch)
                predictions[arm.key].append(prediction.float().cpu())
            logger.info(
                "stage=factorial_decode seed=%s matrix=%s/72 depth=%s role=%s tiles=%s/%s unique_arms=%s elapsed=%.1fs",
                seed,
                record_idx + 1,
                record.depth,
                record.role,
                end,
                tile_count,
                len(COMPUTED_ARMS),
                time.monotonic() - started,
            )

        computed_rows: dict[str, dict[str, Any]] = {}
        for arm in COMPUTED_ARMS:
            prediction, coverage = gate.reassemble(
                torch.cat(predictions[arm.key]),
                tiling,
                tuple(record.W.shape),
            )
            if not torch.all(coverage == 1) or not torch.isfinite(prediction).all():
                raise RuntimeError(f"invalid factorial reconstruction: {seed}/{record.source_key}/{arm.key}")
            computed_rows[arm.key] = metric_row(
                record=record,
                seed=seed,
                arm=arm,
                prediction=prediction,
                num_tiles=tile_count,
            )

        # Expand the four unique zero-code decodes to the rectangular 2x4x4
        # design. The source computed arm and reuse flag remain explicit.
        for enc in ENC_CONDITIONS:
            for code in CODE_CONDITIONS:
                for dec in DEC_CONDITIONS:
                    computed = Arm("none", "zero", dec) if code == "zero" else Arm(enc, code, dec)
                    row = dict(computed_rows[computed.key])
                    row["encoder_condition"] = enc
                    row["code_condition"] = code
                    row["decoder_condition"] = dec
                    row["computed_arm"] = computed.key
                    row["zero_code_reused_across_encoder_labels"] = code == "zero"
                    rows.append(row)
        logger.info(
            "stage=factorial_matrix_complete seed=%s matrix=%s/72 depth=%s role=%s formal_rows=%s",
            seed,
            record_idx + 1,
            record.depth,
            record.role,
            len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(DEC_CONDITIONS),
        )
    return rows


def aggregate_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["tiling_seed"]),
                str(row["encoder_condition"]),
                str(row["code_condition"]),
                str(row["decoder_condition"]),
            )
        ].append(row)
    for (seed, enc, code, dec), values in sorted(grouped.items()):
        scopes: list[tuple[str, list[dict[str, Any]]]] = [("micro", values)]
        scopes.extend((role, [row for row in values if row["role"] == role]) for role in gate.ROLES)
        scope_rows: dict[str, dict[str, Any]] = {}
        for scope, selected in scopes:
            w_target = sum(float(row["w_target"]) for row in selected)
            w_pred = sum(float(row["w_pred"]) for row in selected)
            w_dot = sum(float(row["w_dot"]) for row in selected)
            x_target = sum(float(row["x_target"]) for row in selected)
            x_pred = sum(float(row["x_pred"]) for row in selected)
            x_dot = sum(float(row["x_dot"]) for row in selected)
            w_error = sum(float(row["w_error"]) for row in selected)
            x_error = sum(float(row["x_error"]) for row in selected)
            scope_rows[scope] = {
                "tiling_seed": seed,
                "encoder_condition": enc,
                "code_condition": code,
                "decoder_condition": dec,
                "aggregation": scope,
                "matrices": len(selected),
                "raw_E_W": w_error / w_target,
                "raw_E_X": x_error / x_target,
                "raw_weight_cosine": w_dot / math.sqrt(max(w_target * w_pred, 1e-300)),
                "raw_operator_cosine": x_dot / math.sqrt(max(x_target * x_pred, 1e-300)),
            }
            output.append(scope_rows[scope])
        output.append(
            {
                "tiling_seed": seed,
                "encoder_condition": enc,
                "code_condition": code,
                "decoder_condition": dec,
                "aggregation": "macro",
                "matrices": len(values),
                "raw_E_W": float(np.mean([scope_rows[role]["raw_E_W"] for role in gate.ROLES])),
                "raw_E_X": float(np.mean([scope_rows[role]["raw_E_X"] for role in gate.ROLES])),
                "raw_weight_cosine": float(
                    np.mean([scope_rows[role]["raw_weight_cosine"] for role in gate.ROLES])
                ),
                "raw_operator_cosine": float(
                    np.mean([scope_rows[role]["raw_operator_cosine"] for role in gate.ROLES])
                ),
            }
        )
    return output


def aggregate_sample_scopes(
    indexed: dict[tuple[int, str], dict[str, Any]],
    sampled_depths: np.ndarray,
    *,
    metric: str,
) -> dict[str, np.ndarray]:
    role_numerators: list[np.ndarray] = []
    role_denominators: list[np.ndarray] = []
    prefix = "x" if metric == "E_X" else "w"
    for role in gate.ROLES:
        numerators = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_error"]) for depth in range(12)],
            dtype=np.float64,
        )
        denominators = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_target"]) for depth in range(12)],
            dtype=np.float64,
        )
        role_numerators.append(numerators[sampled_depths].sum(axis=1))
        role_denominators.append(denominators[sampled_depths].sum(axis=1))
    numerator = np.stack(role_numerators, axis=1)
    denominator = np.stack(role_denominators, axis=1)
    role_values = numerator / denominator
    output = {
        role: role_values[:, index]
        for index, role in enumerate(gate.ROLES)
    }
    output["macro"] = role_values.mean(axis=1)
    output["micro"] = numerator.sum(axis=1) / denominator.sum(axis=1)
    return output


def aggregate_sample_cosines(
    indexed: dict[tuple[int, str], dict[str, Any]],
    sampled_depths: np.ndarray,
    *,
    metric: str,
) -> dict[str, np.ndarray]:
    prefix = "x" if metric == "operator_cosine" else "w"
    role_values: list[np.ndarray] = []
    role_dots: list[np.ndarray] = []
    role_targets: list[np.ndarray] = []
    role_predictions: list[np.ndarray] = []
    for role in gate.ROLES:
        dots = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_dot"]) for depth in range(12)],
            dtype=np.float64,
        )[sampled_depths].sum(axis=1)
        targets = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_target"]) for depth in range(12)],
            dtype=np.float64,
        )[sampled_depths].sum(axis=1)
        predictions = np.asarray(
            [float(indexed[(depth, role)][f"{prefix}_pred"]) for depth in range(12)],
            dtype=np.float64,
        )[sampled_depths].sum(axis=1)
        values = dots / np.sqrt(np.maximum(targets * predictions, 1e-300))
        role_values.append(values)
        role_dots.append(dots)
        role_targets.append(targets)
        role_predictions.append(predictions)
    stacked_values = np.stack(role_values, axis=1)
    stacked_dots = np.stack(role_dots, axis=1)
    stacked_targets = np.stack(role_targets, axis=1)
    stacked_predictions = np.stack(role_predictions, axis=1)
    output = {
        role: stacked_values[:, index]
        for index, role in enumerate(gate.ROLES)
    }
    output["macro"] = stacked_values.mean(axis=1)
    output["micro"] = stacked_dots.sum(axis=1) / np.sqrt(
        np.maximum(stacked_targets.sum(axis=1) * stacked_predictions.sum(axis=1), 1e-300)
    )
    return output


def paired_code_effects(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sampled_depths = rng.integers(0, 12, size=(draws, 12))
    # The depth+6 donor is an involution, so its 12 depth effects form six
    # dependent orbits. Resample six orbits and carry both members together.
    sampled_orbits = rng.integers(0, 6, size=(draws, 6))
    sampled_depth_orbits = np.concatenate([sampled_orbits, sampled_orbits + 6], axis=1)
    output: list[dict[str, Any]] = []
    seeds = sorted({int(row["tiling_seed"]) for row in rows})
    for tiling_seed in seeds:
        for enc in ENC_CONDITIONS:
            for dec in DEC_CONDITIONS:
                selected_by_code: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
                for code in CODE_CONDITIONS:
                    selected = [
                        row
                        for row in rows
                        if int(row["tiling_seed"]) == tiling_seed
                        and row["encoder_condition"] == enc
                        and row["code_condition"] == code
                        and row["decoder_condition"] == dec
                    ]
                    selected_by_code[code] = {(int(row["depth"]), str(row["role"])): row for row in selected}
                    if len(selected_by_code[code]) != 72:
                        raise RuntimeError(f"incomplete factorial cell: {tiling_seed}/{enc}/{code}/{dec}")
                for comparator in ("permuted_within_row", "deranged_block", "zero"):
                    comparator_depth_draws = (
                        sampled_depth_orbits if comparator == "deranged_block" else sampled_depths
                    )
                    bootstrap_unit = (
                        "six_depth_plus_6_orbits" if comparator == "deranged_block" else "12_transformer_blocks"
                    )
                    effective_clusters = 6 if comparator == "deranged_block" else 12
                    for metric in ("E_X", "E_W"):
                        correct = aggregate_sample_scopes(
                            selected_by_code["correct"], comparator_depth_draws, metric=metric
                        )
                        reference = aggregate_sample_scopes(
                            selected_by_code[comparator], comparator_depth_draws, metric=metric
                        )
                        exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
                        correct_point = aggregate_sample_scopes(
                            selected_by_code["correct"], exact_depths, metric=metric
                        )
                        reference_point = aggregate_sample_scopes(
                            selected_by_code[comparator], exact_depths, metric=metric
                        )
                        for aggregation in ("macro", "micro", *gate.ROLES):
                            correct_draws = correct[aggregation]
                            reference_draws = reference[aggregation]
                            if (
                                not np.isfinite(correct_draws).all()
                                or not np.isfinite(reference_draws).all()
                                or np.any(correct_draws <= 0)
                                or np.any(reference_draws < 0)
                            ):
                                raise RuntimeError(
                                    f"invalid bootstrap error draws: {tiling_seed}/{enc}/{dec}/{comparator}/"
                                    f"{metric}/{aggregation}"
                                )
                            delta = correct_draws - reference_draws
                            ratio = reference_draws / correct_draws
                            point_delta = float(correct_point[aggregation][0] - reference_point[aggregation][0])
                            point_ratio = float(reference_point[aggregation][0] / correct_point[aggregation][0])
                            ratio_l95 = float(np.quantile(ratio, 0.05))
                            output.append(
                                {
                                    "tiling_seed": tiling_seed,
                                    "encoder_condition": enc,
                                    "decoder_condition": dec,
                                    "metric": metric,
                                    "aggregation": aggregation,
                                    "contrast": f"correct_minus_{comparator}",
                                    "comparator": comparator,
                                    "point_correct": float(correct_point[aggregation][0]),
                                    "point_comparator": float(reference_point[aggregation][0]),
                                    "point_delta_correct_minus_comparator": point_delta,
                                    "delta_l95": float(np.quantile(delta, 0.05)),
                                    "delta_u95": float(np.quantile(delta, 0.95)),
                                    "point_ratio_comparator_over_correct": point_ratio,
                                    "ratio_l95": ratio_l95,
                                    "ratio_u95": float(np.quantile(ratio, 0.95)),
                                    "passes_point_ratio_1p05_and_ratio_l95_gt_1": bool(
                                        point_ratio >= 1.05 and ratio_l95 > 1.0
                                    ),
                                    "draws": draws,
                                    "bootstrap_unit": bootstrap_unit,
                                    "effective_clusters": effective_clusters,
                                    "negative_means_correct_is_better": True,
                                    "ratio_above_one_means_correct_is_better": True,
                                }
                            )
    return output


def paired_code_cosine_effects(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sampled_depths = rng.integers(0, 12, size=(draws, 12))
    sampled_orbits = rng.integers(0, 6, size=(draws, 6))
    sampled_depth_orbits = np.concatenate([sampled_orbits, sampled_orbits + 6], axis=1)
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    output: list[dict[str, Any]] = []
    for tiling_seed in sorted({int(row["tiling_seed"]) for row in rows}):
        for enc in ENC_CONDITIONS:
            for dec in DEC_CONDITIONS:
                selected_by_code: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
                for code in CODE_CONDITIONS:
                    selected = [
                        row
                        for row in rows
                        if int(row["tiling_seed"]) == tiling_seed
                        and row["encoder_condition"] == enc
                        and row["code_condition"] == code
                        and row["decoder_condition"] == dec
                    ]
                    selected_by_code[code] = {
                        (int(row["depth"]), str(row["role"])): row
                        for row in selected
                    }
                    if len(selected_by_code[code]) != 72:
                        raise RuntimeError(f"incomplete cosine factorial cell: {tiling_seed}/{enc}/{code}/{dec}")
                for comparator in ("permuted_within_row", "deranged_block", "zero"):
                    depth_draws = sampled_depth_orbits if comparator == "deranged_block" else sampled_depths
                    bootstrap_unit = (
                        "six_depth_plus_6_orbits" if comparator == "deranged_block" else "12_transformer_blocks"
                    )
                    effective_clusters = 6 if comparator == "deranged_block" else 12
                    for metric in ("operator_cosine", "weight_cosine"):
                        correct = aggregate_sample_cosines(
                            selected_by_code["correct"], depth_draws, metric=metric
                        )
                        reference = aggregate_sample_cosines(
                            selected_by_code[comparator], depth_draws, metric=metric
                        )
                        correct_point = aggregate_sample_cosines(
                            selected_by_code["correct"], exact_depths, metric=metric
                        )
                        reference_point = aggregate_sample_cosines(
                            selected_by_code[comparator], exact_depths, metric=metric
                        )
                        for aggregation in ("macro", "micro", *gate.ROLES):
                            delta = correct[aggregation] - reference[aggregation]
                            if not np.isfinite(delta).all():
                                raise RuntimeError(
                                    f"invalid cosine bootstrap draws: {tiling_seed}/{enc}/{dec}/{comparator}/"
                                    f"{metric}/{aggregation}"
                                )
                            output.append(
                                {
                                    "tiling_seed": tiling_seed,
                                    "encoder_condition": enc,
                                    "decoder_condition": dec,
                                    "metric": metric,
                                    "aggregation": aggregation,
                                    "contrast": f"correct_minus_{comparator}",
                                    "comparator": comparator,
                                    "point_correct": float(correct_point[aggregation][0]),
                                    "point_comparator": float(reference_point[aggregation][0]),
                                    "point_delta_correct_minus_comparator": float(
                                        correct_point[aggregation][0] - reference_point[aggregation][0]
                                    ),
                                    "delta_l95": float(np.quantile(delta, 0.05)),
                                    "delta_u95": float(np.quantile(delta, 0.95)),
                                    "draws": draws,
                                    "bootstrap_unit": bootstrap_unit,
                                    "effective_clusters": effective_clusters,
                                    "positive_means_correct_has_higher_cosine": True,
                                }
                            )
    return output


def paired_c_effects(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sampled_depths = rng.integers(0, 12, size=(draws, 12))
    exact_depths = np.arange(12, dtype=np.int64).reshape(1, 12)
    output: list[dict[str, Any]] = []
    cells = {
        "CC": ("cell", "cell"),
        "NC": ("native", "cell"),
        "CN": ("cell", "native"),
        "NN": ("native", "native"),
    }
    contrasts = {
        "encoder_native_minus_cell_at_cell_decoder": {"NC": 1.0, "CC": -1.0},
        "encoder_native_minus_cell_at_native_decoder": {"NN": 1.0, "CN": -1.0},
        "decoder_native_minus_cell_at_cell_encoder": {"CN": 1.0, "CC": -1.0},
        "decoder_native_minus_cell_at_native_encoder": {"NN": 1.0, "NC": -1.0},
        "interaction_nn_minus_nc_minus_cn_plus_cc": {
            "NN": 1.0,
            "NC": -1.0,
            "CN": -1.0,
            "CC": 1.0,
        },
    }
    for tiling_seed in sorted({int(row["tiling_seed"]) for row in rows}):
        indexed: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
        for cell, (enc, dec) in cells.items():
            selected = [
                row
                for row in rows
                if int(row["tiling_seed"]) == tiling_seed
                and row["encoder_condition"] == enc
                and row["code_condition"] == "correct"
                and row["decoder_condition"] == dec
            ]
            indexed[cell] = {(int(row["depth"]), str(row["role"])): row for row in selected}
            if len(indexed[cell]) != 72:
                raise RuntimeError(f"incomplete correct-code C cell: {tiling_seed}/{cell}")
        for metric in ("E_X", "E_W", "operator_cosine", "weight_cosine"):
            aggregator = aggregate_sample_scopes if metric in {"E_X", "E_W"} else aggregate_sample_cosines
            cell_draws = {
                cell: aggregator(values, sampled_depths, metric=metric)
                for cell, values in indexed.items()
            }
            cell_points = {
                cell: aggregator(values, exact_depths, metric=metric)
                for cell, values in indexed.items()
            }
            for contrast, coefficients in contrasts.items():
                for aggregation in ("macro", "micro", *gate.ROLES):
                    delta = sum(
                        coefficient * cell_draws[cell][aggregation]
                        for cell, coefficient in coefficients.items()
                    )
                    point = sum(
                        coefficient * cell_points[cell][aggregation][0]
                        for cell, coefficient in coefficients.items()
                    )
                    if not np.isfinite(delta).all() or not math.isfinite(float(point)):
                        raise RuntimeError(
                            f"invalid C-effect bootstrap: {tiling_seed}/{metric}/{contrast}/{aggregation}"
                        )
                    output.append(
                        {
                            "tiling_seed": tiling_seed,
                            "metric": metric,
                            "aggregation": aggregation,
                            "contrast": contrast,
                            "coefficients": json.dumps(coefficients, sort_keys=True),
                            "point_contrast": float(point),
                            "l95": float(np.quantile(delta, 0.05)),
                            "u95": float(np.quantile(delta, 0.95)),
                            "draws": draws,
                            "bootstrap_unit": "12_transformer_blocks",
                            "effective_clusters": 12,
                            "positive_means_metric_increases": True,
                        }
                    )
    return output


def parent_baseline_parity(parent: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent_rows = read_csv(parent / "matrix_metrics.csv")
    parent_index = {
        (int(row["tiling_seed"]), str(row["method"]), int(row["depth"]), str(row["role"])): row
        for row in parent_rows
    }
    comparisons = (
        (("cell", "correct", "cell"), "ae_cell_mean_c0"),
        (("native", "correct", "native"), "ae_native_c"),
    )
    details: list[dict[str, Any]] = []
    sufficient_keys = (
        "w_target",
        "w_pred",
        "w_dot",
        "w_error",
        "x_target",
        "x_pred",
        "x_dot",
        "x_error",
        "raw_E_W",
        "raw_E_X",
        "raw_weight_cosine",
        "raw_operator_cosine",
        "raw_weight_norm_ratio",
        "raw_operator_norm_ratio",
    )
    for (enc, code, dec), method in comparisons:
        selected = [
            row
            for row in rows
            if row["encoder_condition"] == enc
            and row["code_condition"] == code
            and row["decoder_condition"] == dec
        ]
        for row in selected:
            parent_row = parent_index[(int(row["tiling_seed"]), method, int(row["depth"]), str(row["role"]))]
            absolute = {
                key: abs(float(row[key]) - float(parent_row[key]))
                for key in sufficient_keys
            }
            relative = {
                key: absolute[key] / max(abs(float(parent_row[key])), 1e-300)
                for key in sufficient_keys
            }
            details.append(
                {
                    "tiling_seed": int(row["tiling_seed"]),
                    "depth": int(row["depth"]),
                    "role": str(row["role"]),
                    "factorial_arm": f"{enc}/{code}/{dec}",
                    "parent_method": method,
                    "raw_E_X_abs_diff": abs(float(row["raw_E_X"]) - float(parent_row["raw_E_X"])),
                    "raw_E_W_abs_diff": abs(float(row["raw_E_W"]) - float(parent_row["raw_E_W"])),
                    "prediction_sha256_equal": row["prediction_sha256"] == parent_row["prediction_sha256"],
                    "max_sufficient_stat_abs_diff": max(absolute.values()),
                    "max_sufficient_stat_rel_diff": max(relative.values()),
                }
            )
    max_ex = max(float(row["raw_E_X_abs_diff"]) for row in details)
    max_ew = max(float(row["raw_E_W_abs_diff"]) for row in details)
    prediction_hash_equal = sum(bool(row["prediction_sha256_equal"]) for row in details)
    prediction_hash_total = len(details)
    max_stat_abs = max(float(row["max_sufficient_stat_abs_diff"]) for row in details)
    max_stat_rel = max(float(row["max_sufficient_stat_rel_diff"]) for row in details)
    return {
        "pass": (
            prediction_hash_equal == prediction_hash_total
            and max_stat_abs <= 1e-9
            and max_stat_rel <= 1e-12
        ),
        "prediction_hash_gate": "all 288 parent/factorial predictions must have identical SHA256",
        "sufficient_stat_abs_tolerance": 1e-9,
        "sufficient_stat_rel_tolerance": 1e-12,
        "max_raw_E_X_abs_diff": max_ex,
        "max_raw_E_W_abs_diff": max_ew,
        "max_sufficient_stat_abs_diff": max_stat_abs,
        "max_sufficient_stat_rel_diff": max_stat_rel,
        "prediction_sha256_equal": prediction_hash_equal,
        "prediction_sha256_total": prediction_hash_total,
        "details": details,
    }


def zero_reuse_invariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[int, int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["code_condition"] != "zero":
            continue
        indexed[
            (
                int(row["tiling_seed"]),
                int(row["depth"]),
                str(row["role"]),
                str(row["decoder_condition"]),
                str(row["encoder_condition"]),
            )
        ] = row
    mismatches: list[dict[str, Any]] = []
    for seed in sorted({key[0] for key in indexed}):
        for depth in range(12):
            for role in gate.ROLES:
                for dec in DEC_CONDITIONS:
                    left = indexed[(seed, depth, role, dec, "cell")]
                    right = indexed[(seed, depth, role, dec, "native")]
                    if left["prediction_sha256"] != right["prediction_sha256"]:
                        mismatches.append(
                            {"tiling_seed": seed, "depth": depth, "role": role, "decoder_condition": dec}
                        )
    return {
        "pass": not mismatches,
        "mismatches": mismatches,
        "pairs_checked": len({key[:4] for key in indexed}),
    }


def validate_factorial_grid(rows: list[dict[str, Any]], tiling_seeds: tuple[int, ...]) -> dict[str, Any]:
    expected = {
        (seed, depth, role, enc, code, dec)
        for seed in tiling_seeds
        for depth in range(12)
        for role in gate.ROLES
        for enc in ENC_CONDITIONS
        for code in CODE_CONDITIONS
        for dec in DEC_CONDITIONS
    }
    observed = [
        (
            int(row["tiling_seed"]),
            int(row["depth"]),
            str(row["role"]),
            str(row["encoder_condition"]),
            str(row["code_condition"]),
            str(row["decoder_condition"]),
        )
        for row in rows
    ]
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != expected:
        raise RuntimeError(
            f"factorial grid duplicate/missing rows: observed={len(observed)} unique={len(set(observed))} "
            f"expected={len(expected)}"
        )
    if any(
        str(row["shape"]) != "x".join(str(value) for value in gate.ROLE_SHAPES[str(row["role"])])
        or int(row["num_tiles"])
        != (gate.ROLE_SHAPES[str(row["role"])][0] // 64)
        * (gate.ROLE_SHAPES[str(row["role"])][1] // 64)
        for row in rows
    ):
        raise RuntimeError("factorial row has wrong role geometry or nonpositive tile count")

    by_matrix: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_matrix[(int(row["tiling_seed"]), int(row["depth"]), str(row["role"]))].append(row)
    collisions: list[dict[str, Any]] = []
    for matrix_key, values in sorted(by_matrix.items()):
        unique_computed: dict[str, dict[str, Any]] = {}
        for row in values:
            computed = str(row["computed_arm"])
            if computed in unique_computed:
                if row["code_condition"] != "zero":
                    raise RuntimeError(f"non-zero computed arm duplicated: {matrix_key}/{computed}")
                continue
            unique_computed[computed] = row
        if len(unique_computed) != len(COMPUTED_ARMS):
            raise RuntimeError(f"computed arm count mismatch: {matrix_key}/{len(unique_computed)}")
        by_hash: dict[str, list[str]] = defaultdict(list)
        for computed, row in unique_computed.items():
            by_hash[str(row["prediction_sha256"])].append(computed)
        for prediction_sha, arms in by_hash.items():
            if len(arms) > 1:
                collisions.append(
                    {
                        "tiling_seed": matrix_key[0],
                        "depth": matrix_key[1],
                        "role": matrix_key[2],
                        "prediction_sha256": prediction_sha,
                        "computed_arms": sorted(arms),
                    }
                )
    return {
        "pass": True,
        "formal_rows": len(observed),
        "formal_unique_keys": len(set(observed)),
        "formal_arms_per_matrix": len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(DEC_CONDITIONS),
        "unique_computed_arms_per_matrix": len(COMPUTED_ARMS),
        "prediction_collision_free": not collisions,
        "prediction_collisions_require_human_investigation": collisions,
    }


def mechanism_screen(
    effect_rows: list[dict[str, Any]],
    cosine_effect_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def robust(seed: int, enc: str, dec: str, comparator: str) -> bool:
        required = [
            row
            for row in effect_rows
            if int(row["tiling_seed"]) == seed
            and row["encoder_condition"] == enc
            and row["decoder_condition"] == dec
            and row["metric"] == "E_X"
            and row["contrast"] == f"correct_minus_{comparator}"
            and row["aggregation"] in {"macro", "micro"}
        ]
        return len(required) == 2 and all(
            float(row["point_ratio_comparator_over_correct"]) >= 1.05
            and float(row["ratio_l95"]) > 1.0
            and bool(row["passes_point_ratio_1p05_and_ratio_l95_gt_1"])
            for row in required
        )

    seeds = sorted({int(row["tiling_seed"]) for row in effect_rows})
    by_tiling = {
        str(seed): {
            "cell_cell_correct_better_than_within_row_permutation": robust(
                seed, "cell", "cell", "permuted_within_row"
            ),
            "cell_cell_correct_better_than_zero": robust(seed, "cell", "cell", "zero"),
            "native_native_correct_better_than_within_row_permutation": robust(
                seed, "native", "native", "permuted_within_row"
            ),
            "native_native_correct_better_than_zero": robust(seed, "native", "native", "zero"),
            "secondary_cell_cell_correct_better_than_wrong_block": robust(
                seed, "cell", "cell", "deranged_block"
            ),
            "cell_cell_correct_has_higher_operator_cosine_than_within_row": all(
                float(row["delta_l95"]) > 0.0
                for row in cosine_effect_rows
                if int(row["tiling_seed"]) == seed
                and row["encoder_condition"] == "cell"
                and row["decoder_condition"] == "cell"
                and row["metric"] == "operator_cosine"
                and row["comparator"] == "permuted_within_row"
                and row["aggregation"] in {"macro", "micro"}
            )
            and sum(
                1
                for row in cosine_effect_rows
                if int(row["tiling_seed"]) == seed
                and row["encoder_condition"] == "cell"
                and row["decoder_condition"] == "cell"
                and row["metric"] == "operator_cosine"
                and row["comparator"] == "permuted_within_row"
                and row["aggregation"] in {"macro", "micro"}
            )
            == 2,
        }
        for seed in seeds
    }
    fixed_checks = [
        value
        for seed_rows in by_tiling.values()
        for key, value in seed_rows.items()
        if key.startswith("cell_cell_")
        and key != "cell_cell_correct_has_higher_operator_cosine_than_within_row"
    ]
    native_checks = [
        value
        for seed_rows in by_tiling.values()
        for key, value in seed_rows.items()
        if key.startswith("native_native_")
    ]
    directional_checks = [
        bool(rows["cell_cell_correct_has_higher_operator_cosine_than_within_row"])
        for rows in by_tiling.values()
    ]
    fixed_rows = [
        row
        for row in effect_rows
        if row["encoder_condition"] == "cell"
        and row["decoder_condition"] == "cell"
        and row["metric"] == "E_X"
        and row["comparator"] in {"permuted_within_row", "zero"}
        and row["aggregation"] in {"macro", "micro"}
    ]
    if len(fixed_rows) != 8:
        raise RuntimeError(f"fixed-C primary screen requires exactly 8 rows, got {len(fixed_rows)}")
    exclusion_rows = [
        {
            "tiling_seed": int(row["tiling_seed"]),
            "comparator": str(row["comparator"]),
            "aggregation": str(row["aggregation"]),
            "ratio_u95": float(row["ratio_u95"]),
            "excludes_ratio_at_least_1p05": float(row["ratio_u95"]) < 1.05,
        }
        for row in fixed_rows
    ]
    primary_pass = bool(all(fixed_checks))
    native_pass = bool(all(native_checks))
    directional_pass = bool(all(directional_checks))
    any_exclusion = any(bool(row["excludes_ratio_at_least_1p05"]) for row in exclusion_rows)
    if primary_pass:
        primary_status = "SUPPORTED_ON_FIXED_C_SOURCE_PANEL"
    elif any_exclusion:
        primary_status = "PREREGISTERED_UNIVERSAL_8_ROW_CLAIM_EXCLUDED_AT_ONE_OR_MORE_ENDPOINTS"
    else:
        primary_status = "INCONCLUSIVE_FOR_5_PERCENT_EFFECT"
    return {
        "primary_fixed_c_criterion": (
            "for raw E_X, point comparator/correct ratio >=1.05 and paired one-sided ratio L95 >1 "
            "for macro and micro, within-row and zero controls, and both tilings (8 jointly required rows)"
        ),
        "native_native_robustness_criterion": "the analogous 8 rows under native/native context",
        "context_robust_criterion": "all 16 fixed-C primary plus native/native robustness rows",
        "directional_operator_structure_criterion": (
            "under cell/cell, correct-minus-within-row operator cosine paired L95 > 0 for macro and micro "
            "on both tilings; required before directional/operator-structure wording"
        ),
        "primary_wrong_code_control": "same-matrix same-row-group no-fixed-point column-code permutation",
        "secondary_wrong_layer_control": "depth+6 same-role donor with six-orbit cluster bootstrap",
        "by_tiling": by_tiling,
        "primary_fixed_c_load_bearing_pass": primary_pass,
        "native_native_robustness_pass": native_pass,
        "strong_context_robust_all_16_pass": bool(primary_pass and native_pass),
        "directional_operator_cosine_pass": directional_pass,
        "fixed_c_tile_specific_with_directional_structure_pass": bool(primary_pass and directional_pass),
        "strong_tile_specific_code_under_fixed_c": primary_pass,
        "strong_tile_specific_code_under_native_c": native_pass,
        "strong_weight_code_dependence_screen": primary_pass,
        "primary_fixed_c_status": primary_status,
        "fixed_c_5pct_exclusion_rows": exclusion_rows,
        "fixed_c_universal_claim_excluded_by_any_endpoint": any_exclusion,
        "interpretation_if_not_pass": (
            "non-pass is not automatically falsification: an endpoint with ratio U95 <1.05 excludes the "
            "preregistered >=5% effect there; otherwise the result is inconclusive. Smaller, role-local, or "
            "tiling-specific effects are exploratory and cannot support the main claim"
        ),
        "scope_limit": (
            "a pass establishes a causal W/tile-specific z contribution on this source panel, not good absolute "
            "reconstruction, cross-domain transfer, or a global model-weight manifold"
        ),
    }


def main() -> None:
    args = parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.bootstrap_draws <= 0:
        raise ValueError("bootstrap-draws must be positive")
    output = assert_output_allowlisted(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be fresh and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    target_seal = install_target_access_seal()
    argv_contains_target_marker = any(
        marker in " ".join(sys.argv).lower()
        for marker in target_seal["forbidden_path_markers"]
    )
    if argv_contains_target_marker:
        raise RuntimeError("source-only command line contains a forbidden target marker")
    gate.write_json(
        output / "target_access_seal.json",
        {
            **target_seal,
            "status": "ACTIVE_BEFORE_PARENT_OR_HELDOUT_READ",
            "target_data2vec_access": False,
            "argv_contains_target_marker": argv_contains_target_marker,
        },
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", mode="w")],
        force=True,
    )
    logger = logging.getLogger("source_latent_code_factorial")
    started = time.monotonic()
    random.seed(args.factorial_seed)
    np.random.seed(args.factorial_seed)
    torch.manual_seed(args.factorial_seed)

    logger.info("stage=parent_artifact_audit")
    parent = require_exact_realpath(args.parent_run_dir, EXPECTED_PARENT_REALPATH, label="parent run")
    audit = audit_parent(parent)
    logger.info("stage=template_snapshot_audit")
    templates, template_distances, template_snapshot = validate_template_snapshot(parent, audit)
    write_preexecution_contract(args=args, output=output, audit=audit)
    gate.write_csv(output / "decoder_template_distance_diagnostics.csv", template_distances)
    gate.write_json(output / "template_snapshot.json", template_snapshot)
    logger.info(
        "resolved_config parent=%s output=%s device=%s dtype=parent_amp seed=%s batch=%s "
        "cache_mode=locked_parent_source_only execute=%s target_access=false",
        parent,
        output,
        args.device,
        args.factorial_seed,
        args.batch_size,
        args.execute,
    )
    logger.info("stage=heldout_panel_load source_realpath=%s", audit["heldout_realpath"])
    records = load_records(parent, audit, int(audit["resolved_config"]["seed"]), logger)
    logger.info("stage=locked_tiling_cpu_revalidation")
    tiling_rows = validate_locked_tilings(records, audit)
    gate.write_csv(output / "locked_tiling_revalidation.csv", tiling_rows)
    logger.info("stage=global_derangement_manifest")
    derangement_rows = build_code_derangement_manifest(records, audit["tiling_seeds"])
    gate.write_csv(output / "code_derangement_manifest.csv", derangement_rows)
    derangement_validity = validate_code_derangement_manifest(derangement_rows, audit["tiling_seeds"])
    gate.write_json(output / "code_derangement_validity.json", derangement_validity)
    recheck_parent_immutable(parent, audit)
    preexecution = read_json(output / "preexecution_contract.json")
    preexecution["source_cpu_preflight_lock"] = {
        "status": "LOCKED_EXACT_SOURCE_PANEL_TEMPLATES_TILINGS_DERANGEMENTS_NO_MODEL_FORWARD",
        "template_snapshot_sha256": gate.sha256_file(output / "template_snapshot.json"),
        "template_distance_sha256": gate.sha256_file(output / "decoder_template_distance_diagnostics.csv"),
        "tiling_revalidation_sha256": gate.sha256_file(output / "locked_tiling_revalidation.csv"),
        "derangement_manifest_sha256": gate.sha256_file(output / "code_derangement_manifest.csv"),
        "derangement_validity_sha256": gate.sha256_file(output / "code_derangement_validity.json"),
        "tiling_rows": len(tiling_rows),
        "derangement_rows": len(derangement_rows),
        "tiles_per_tiling": {
            str(seed): sum(int(row["num_tiles"]) for row in tiling_rows if int(row["tiling_seed"]) == seed)
            for seed in audit["tiling_seeds"]
        },
        "target_data2vec_access": False,
    }
    gate.write_json(output / "preexecution_contract.json", preexecution)
    if not args.execute:
        gate.write_json(
            output / "audit_only_status.json",
            {
                "status": "AUDIT_ONLY_COMPLETE_NO_WEIGHT_AE_OR_DISTRIBUTION_ENCODER_FORWARD",
                "elapsed_seconds": time.monotonic() - started,
                "next_command_requires_explicit_execute": True,
                "exact_parent_hashes_pass": True,
                "exact_heldout_vit_b_realpath_and_model_pass": True,
                "template_tensor_recheck_pass": True,
                "locked_tiling_cpu_revalidation_pass": True,
                "global_derangement_manifest_pass": True,
                "target_access_seal_sha256": gate.sha256_file(output / "target_access_seal.json"),
                "target_data2vec_access": False,
            },
        )
        logger.info("completed audit-only; no Weight-AE/distribution-encoder forward")
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.factorial_seed)
    logger.info("stage=checkpoint_load")
    checkpoint_path = Path(audit["checkpoint_info"]["path"])
    model, amp_enabled, amp_dtype, model_contract = build_model(
        checkpoint_path=checkpoint_path,
        device=device,
        no_amp=args.no_amp,
    )
    gate.write_json(output / "model_path_contract.json", model_contract)
    logger.info("model_path_contract=%s", json.dumps(model_contract, sort_keys=True, default=str))

    logger.info("stage=runtime_immutable_recheck_before_first_model_forward")
    recheck_parent_immutable(parent, audit)
    runtime_templates, runtime_distances, runtime_template_snapshot = validate_template_snapshot(parent, audit)
    if runtime_template_snapshot != template_snapshot or any(
        gate.tensor_sha256(runtime_templates["cell_mean"][cell][name])
        != gate.tensor_sha256(templates["cell_mean"][cell][name])
        for cell in templates["cell_mean"]
        for name in ("c_var", "c_patch")
    ):
        raise RuntimeError("loaded source template tensors changed before first model forward")
    if runtime_distances != template_distances:
        raise RuntimeError("decoder-template distance snapshot changed before first model forward")
    assert_no_target_modules()
    if TARGET_ACCESS_SEAL.get("installed") is not True:
        raise RuntimeError("source-only target access seal was not installed")
    preexecution = read_json(output / "preexecution_contract.json")
    preexecution["runtime_lock"] = {
        "status": "LOCKED_PARENT_PANEL_TEMPLATES_TILINGS_BEFORE_FIRST_FACTORIAL_WEIGHT_FORWARD",
        "model_contract_sha256": gate.sha256_file(output / "model_path_contract.json"),
        "tiling_revalidation_sha256": gate.sha256_file(output / "locked_tiling_revalidation.csv"),
        "template_sha256_rechecked": gate.sha256_file(parent / "source_condition_templates.pt"),
    }
    gate.write_json(output / "preexecution_contract.json", preexecution)

    logger.info("stage=decoder_path_preflight")
    preflight = run_path_preflight(
        model=model,
        records=records,
        templates=templates,
        tiling_seed=audit["tiling_seeds"][0],
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    gate.write_json(output / "decoder_path_preflight.json", preflight)
    if not preflight["pass"]:
        raise RuntimeError(f"decoder-path preflight failed: {preflight}")

    all_rows: list[dict[str, Any]] = []
    all_code_summary: list[dict[str, Any]] = []
    all_latent_pair_summary: list[dict[str, Any]] = []
    all_latent_pair_tiles: list[dict[str, Any]] = []
    all_native_row_diagnostics: list[dict[str, Any]] = []
    for seed in audit["tiling_seeds"]:
        logger.info("stage=seed_code_cache seed=%s", seed)
        (
            codes,
            native_patches,
            summary,
            latent_pair_summary,
            latent_pair_tiles,
            native_row_diagnostics,
        ) = build_seed_cache(
            seed=seed,
            records=records,
            templates=templates,
            model=model,
            device=device,
            batch_size=args.batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            logger=logger,
        )
        all_code_summary.extend(summary)
        all_latent_pair_summary.extend(latent_pair_summary)
        all_latent_pair_tiles.extend(latent_pair_tiles)
        all_native_row_diagnostics.extend(native_row_diagnostics)
        if args.persist_code_cache:
            cache_path = output / f"factorial_code_cache_seed_{seed}.pt"
            torch.save(
                {
                    "checkpoint_sha256": gate.EXPECTED_CHECKPOINT_SHA256,
                    "tiling_seed": seed,
                    "codes": codes,
                    "native_c_patch": native_patches,
                },
                cache_path,
            )
            logger.info("stage=cache_write seed=%s artifact=%s", seed, cache_path)

        logger.info("stage=seed_factorial_decode seed=%s unique_arms=%s", seed, len(COMPUTED_ARMS))
        seed_rows = decode_seed_factorial(
            seed=seed,
            records=records,
            templates=templates,
            codes=codes,
            native_patches=native_patches,
            model=model,
            device=device,
            batch_size=args.batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            logger=logger,
        )
        all_rows.extend(seed_rows)
        del codes, native_patches

    expected_rows = len(audit["tiling_seeds"]) * 72 * len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(DEC_CONDITIONS)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"factorial row count mismatch: expected={expected_rows} got={len(all_rows)}")
    if len(all_native_row_diagnostics) != len(audit["tiling_seeds"]) * 72:
        raise RuntimeError(
            f"native row-condition invariant census mismatch: {len(all_native_row_diagnostics)}"
        )
    numeric_keys = (
        "w_target",
        "w_pred",
        "w_dot",
        "w_error",
        "x_target",
        "x_pred",
        "x_dot",
        "x_error",
        "raw_E_W",
        "raw_E_X",
        "raw_weight_cosine",
        "raw_operator_cosine",
    )
    if not all(math.isfinite(float(row[key])) for row in all_rows for key in numeric_keys):
        raise RuntimeError("non-finite factorial sufficient statistic")

    logger.info("stage=output_sufficient_statistics rows=%s", len(all_rows))
    gate.write_csv(output / "factorial_matrix_sufficient_stats.csv", all_rows)
    gate.write_csv(output / "factorial_code_summary.csv", all_code_summary)
    gate.write_csv(output / "latent_correct_donor_summary.csv", all_latent_pair_summary)
    gate.write_csv(output / "latent_correct_donor_tiles.csv", all_latent_pair_tiles)
    gate.write_csv(output / "native_row_condition_invariance.csv", all_native_row_diagnostics)
    aggregate = aggregate_group(all_rows)
    gate.write_csv(output / "factorial_aggregate_metrics.csv", aggregate)
    effects = paired_code_effects(
        all_rows,
        draws=args.bootstrap_draws,
        seed=args.factorial_seed + 99,
    )
    gate.write_csv(output / "paired_code_effects.csv", effects)
    cosine_effects = paired_code_cosine_effects(
        all_rows,
        draws=args.bootstrap_draws,
        seed=args.factorial_seed + 99,
    )
    gate.write_csv(output / "paired_code_cosine_effects.csv", cosine_effects)
    c_effects = paired_c_effects(
        all_rows,
        draws=args.bootstrap_draws,
        seed=args.factorial_seed + 199,
    )
    gate.write_csv(output / "paired_c_effects.csv", c_effects)

    logger.info("stage=validity_review")
    parity = parent_baseline_parity(parent, all_rows)
    zero_invariance = zero_reuse_invariance(all_rows)
    grid_validity = validate_factorial_grid(all_rows, audit["tiling_seeds"])
    gate.write_json(output / "parent_baseline_parity.json", parity)
    gate.write_json(output / "zero_code_reuse_invariance.json", zero_invariance)
    gate.write_json(output / "factorial_grid_validity.json", grid_validity)
    if not parity["pass"] or not zero_invariance["pass"] or not grid_validity["pass"]:
        raise RuntimeError(
            f"post-factorial validity failure: parity={parity} zero={zero_invariance} grid={grid_validity}"
        )

    screen = mechanism_screen(effects, cosine_effects)
    gate.write_json(output / "mechanism_screen.json", screen)
    artifacts = sorted(
        (
            {
                "path": str(path.resolve()),
                "sha256": gate.sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output.iterdir()
            if path.is_file() and path.suffix in {".csv", ".json", ".pt"}
        ),
        key=lambda row: row["path"],
    )
    gate.write_json(
        output / "run_manifest.json",
        {
            "status": "COMPLETE_SOURCE_ONLY_FACTORIAL_AWAITING_FROZEN_POSTRUN_ANALYZER_AND_PLOT_REVIEW",
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": model_contract,
            "target_access": False,
            "target_data2vec_access": False,
            "target_access_seal": target_seal,
            "counts": {
                "heldout_matrices": 72,
                "tiling_seeds": 2,
                "formal_arms": len(ENC_CONDITIONS) * len(CODE_CONDITIONS) * len(DEC_CONDITIONS),
                "unique_decoder_arms": len(COMPUTED_ARMS),
                "matrix_rows": len(all_rows),
                "latent_pair_summary_rows": len(all_latent_pair_summary),
                "latent_pair_tile_rows": len(all_latent_pair_tiles),
                "native_row_condition_rows": len(all_native_row_diagnostics),
                "bootstrap_draws": args.bootstrap_draws,
                "paired_code_effect_rows": len(effects),
                "paired_code_cosine_effect_rows": len(cosine_effects),
                "paired_c_effect_rows": len(c_effects),
            },
            "validity": {
                "decoder_path_preflight": preflight["pass"],
                "parent_baseline_parity": parity["pass"],
                "zero_code_reuse_invariance": zero_invariance["pass"],
                "factorial_grid": grid_validity["pass"],
            },
            "postrun_analyzer": {
                "path": str(POSTRUN_ANALYZER),
                "sha256": gate.sha256_file(POSTRUN_ANALYZER),
                "required_before_experiment_complete": True,
            },
            "mechanism_screen": screen,
            "artifacts": artifacts,
        },
    )
    logger.info(
        "completed status=AWAITING_FROZEN_POSTRUN_ANALYZER elapsed=%.1fs strong_weight_code_screen=%s artifacts=%s",
        time.monotonic() - started,
        screen["strong_weight_code_dependence_screen"],
        output,
    )


if __name__ == "__main__":
    main()
