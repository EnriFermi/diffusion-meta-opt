#!/usr/bin/env python3
"""Post-hoc direct-weight lineage diagnostic for the global-context ablation.

This script never imports or forwards the Weight-AE.  It reconstructs the exact
72-matrix grids sealed by the formal raw runner, verifies every tensor against
that runner's input audit, and reports descriptive pairwise similarities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path("/home/coder/project")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/crossmodal_united_structure"
RAW_ROOT = ARTIFACT_ROOT / "global_context_latent_geometry_ablation_run_20260816"
SOURCE_PARENT = ARTIFACT_ROOT / "source_confirmatory_gate_20260816_clean2"
SOURCE_HELDOUT = SOURCE_PARENT / "heldout_panel_manifest.json"
SOURCE_HELDOUT_SHA256 = "10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985"
SOURCE_OFFLINE = (
    PROJECT_ROOT
    / "projects/weight-vae/workspace/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset"
)
PANEL_ROOT = ARTIFACT_ROOT / "prospective_geometry_matched_panels_20260816/panels"
PANEL_PATHS = {
    "beans": PANEL_ROOT / "beans/panel.pt",
    "trocr_sroie": PANEL_ROOT / "trocr_sroie/panel.pt",
}
PANEL_SHA256 = {
    "beans": "a47ddd182f09a56fc42d0fe887d0f2717b1972d46afbd1c92c80be13bbc9e6b6",
    "trocr_sroie": "ded6e7a84c1b34f1fe2d7fd1bdf6ccf041df23c1e20a710197815e0bf99f91a9",
}
DEFAULT_OUTPUT = ARTIFACT_ROOT / "global_context_weight_lineage_posthoc_20260816"
ROLES = ("attn_query", "attn_key", "attn_value", "attn_output", "ffn_up", "ffn_down")
PANELS = ("source_vit_b_flickr", "beans", "trocr_sroie")
PAIRS = (
    ("source_vit_b_flickr", "beans"),
    ("source_vit_b_flickr", "trocr_sroie"),
    ("beans", "trocr_sroie"),
)


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


def weight_shape_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError(f"weight must be 2-D: {tuple(value.shape)}")
    array = value.numpy().astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", int(value.shape[0]), int(value.shape[1])))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def matrix_key(depth: int, role: str) -> str:
    return f"depth={depth:02d}|role={role}"


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "maximum": ordered[-1],
    }


def pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a = left.reshape(-1).to(torch.float64)
    b = right.reshape(-1).to(torch.float64)
    left_norm = torch.linalg.vector_norm(a)
    right_norm = torch.linalg.vector_norm(b)
    delta_norm = torch.linalg.vector_norm(a - b)
    values = {
        "cosine": float(torch.dot(a, b) / (left_norm * right_norm)),
        "relative_l2_to_left": float(delta_norm / left_norm),
        "relative_l2_to_right": float(delta_norm / right_norm),
        "symmetric_relative_l2": float(2.0 * delta_norm / (left_norm + right_norm)),
        "right_to_left_norm_ratio": float(right_norm / left_norm),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("non-finite direct-weight similarity")
    return values


def load_exact_grids(raw_audit: Mapping[str, Any], log: logging.Logger) -> dict[str, dict[str, torch.Tensor]]:
    expected_rows = raw_audit["w_only_materialization"]["w_grid_rows"]
    expected = {
        (str(row["panel_id"]), int(row["depth"]), str(row["role"])): row
        for row in expected_rows
    }
    if len(expected) != 216:
        raise RuntimeError(f"raw input-audit W grid has {len(expected)} rows, expected 216")

    if sha256_file(SOURCE_HELDOUT) != SOURCE_HELDOUT_SHA256:
        raise RuntimeError("source heldout manifest hash mismatch")
    heldout = json.loads(SOURCE_HELDOUT.read_text(encoding="utf-8"))
    source_catalog_path = SOURCE_OFFLINE / "sources.json"
    catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    sources = {str(row["source_key"]): row for row in catalog["sources"]}
    grids: dict[str, dict[str, torch.Tensor]] = {panel: {} for panel in PANELS}

    log.info("stage=load_source_weights rows=%d", len(heldout))
    for index, item in enumerate(heldout, start=1):
        ref = item["context_a"]
        depth, role = int(ref["depth"]), str(ref["role"])
        key = matrix_key(depth, role)
        source = sources[str(ref["source_key"])]
        path = (SOURCE_OFFLINE / str(source["weight_path"])).resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        weight = payload["weight"].detach().cpu().to(torch.float32).contiguous()
        grids["source_vit_b_flickr"][key] = weight
        if index % 12 == 0:
            log.info("stage=load_source_weights progress=%d/%d", index, len(heldout))

    for panel, path in PANEL_PATHS.items():
        log.info("stage=load_panel panel=%s path=%s cache=mmap", panel, path)
        if sha256_file(path) != PANEL_SHA256[panel]:
            raise RuntimeError(f"panel hash mismatch: {panel}")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("schema_version") != "prospective_panel_v1" or payload.get("panel_id") != panel:
            raise RuntimeError(f"panel header mismatch: {panel}")
        grids[panel] = {
            str(key): value.detach().cpu().to(torch.float32).contiguous()
            for key, value in payload["weights"].items()
        }

    expected_keys = {matrix_key(depth, role) for depth in range(12) for role in ROLES}
    log.info("stage=verify_exact_raw_grid rows=216")
    for panel in PANELS:
        if set(grids[panel]) != expected_keys:
            raise RuntimeError(f"incomplete matrix grid: {panel}")
        for depth in range(12):
            for role in ROLES:
                weight = grids[panel][matrix_key(depth, role)]
                contract = expected[(panel, depth, role)]
                if (
                    tuple(weight.shape) != tuple(contract["shape"])
                    or not bool(torch.isfinite(weight).all())
                    or tensor_sha256(weight) != contract["weight_tensor_sha256"]
                    or weight_shape_bytes_sha256(weight) != contract["weight_shape_bytes_sha256"]
                ):
                    raise RuntimeError(f"raw input-audit tensor mismatch: {panel}/{depth}/{role}")
    log.info("stage=verify_exact_raw_grid status=PASS")
    return grids


def run(output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    log_path = stage / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("weight_lineage_posthoc")
    try:
        log.info(
            "experiment=global_context_weight_lineage_posthoc device=cpu dtype=FP32_inputs_FP64_metrics "
            "seed=none cache=mmap output_dir=%s scope=POSTHOC_DESCRIPTIVE",
            output_dir,
        )
        raw_manifest_sha = sha256_file(RAW_ROOT / "artifact_manifest.json")
        raw_audit = json.loads((RAW_ROOT / "input_audit.json").read_text(encoding="utf-8"))
        if not raw_audit.get("pass"):
            raise RuntimeError("formal raw input audit is not PASS")
        grids = load_exact_grids(raw_audit, log)

        rows: list[dict[str, Any]] = []
        log.info("stage=pairwise_metrics pairs=%d matrices_per_pair=72", len(PAIRS))
        for left_panel, right_panel in PAIRS:
            for depth in range(12):
                for role in ROLES:
                    key = matrix_key(depth, role)
                    rows.append(
                        {
                            "left_panel": left_panel,
                            "right_panel": right_panel,
                            "depth": depth,
                            "role": role,
                            "num_parameters": grids[left_panel][key].numel(),
                            **pair_metrics(grids[left_panel][key], grids[right_panel][key]),
                        }
                    )
            log.info("stage=pairwise_metrics pair=%s__%s progress=72/72", left_panel, right_panel)

        metric_names = (
            "cosine",
            "relative_l2_to_left",
            "relative_l2_to_right",
            "symmetric_relative_l2",
            "right_to_left_norm_ratio",
        )
        csv_path = stage / "matched_weight_similarity.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary: dict[str, Any] = {
            "schema_version": "global_context_weight_lineage_posthoc_v1",
            "scope": "POSTHOC_DESCRIPTIVE_NOT_A_REGISTERED_DECISION",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_binding": {
                "formal_raw_manifest_sha256": raw_manifest_sha,
                "formal_raw_input_audit_sha256": sha256_file(RAW_ROOT / "input_audit.json"),
                "source_heldout_manifest_sha256": sha256_file(SOURCE_HELDOUT),
                "source_catalog_sha256": sha256_file(SOURCE_OFFLINE / "sources.json"),
                "panel_sha256": {panel: sha256_file(path) for panel, path in PANEL_PATHS.items()},
                "exact_tensor_contract_rows_checked": 216,
            },
            "row_count": len(rows),
            "pairs": {},
        }
        for left_panel, right_panel in PAIRS:
            label = f"{left_panel}__{right_panel}"
            selected = [
                row for row in rows if row["left_panel"] == left_panel and row["right_panel"] == right_panel
            ]
            summary["pairs"][label] = {
                "all_72": {name: describe([float(row[name]) for row in selected]) for name in metric_names},
                "by_role": {
                    role: {
                        name: describe([float(row[name]) for row in selected if row["role"] == role])
                        for name in metric_names
                    }
                    for role in ROLES
                },
            }
        json_dump(stage / "summary.json", summary)

        sb = summary["pairs"]["source_vit_b_flickr__beans"]["all_72"]
        st = summary["pairs"]["source_vit_b_flickr__trocr_sroie"]["all_72"]
        readme = f"""# Post-hoc direct-weight lineage diagnostic

Status: **descriptive post-hoc diagnostic; not a registered decision and not a population claim**.

All 216 tensors were reconstructed from the exact sources used by the formal runner and matched its stored shape-aware and tensor hashes before comparison. Metrics use corresponding role/depth matrices and FP64 reductions.

## Main observation

- Source vs Beans: median cosine `{sb['cosine']['median']:.9f}` (range `{sb['cosine']['minimum']:.9f}`–`{sb['cosine']['maximum']:.9f}`), median relative L2 to source `{sb['relative_l2_to_left']['median']:.9f}`.
- Source vs TrOCR: median cosine `{st['cosine']['median']:.9f}` (range `{st['cosine']['minimum']:.9f}`–`{st['cosine']['maximum']:.9f}`), median relative L2 to source `{st['relative_l2_to_left']['median']:.9f}`.

Thus source and Beans are extremely close at the raw-weight level, while TrOCR is not. In the completed three-panel analysis, source and Beans must not be treated as two independent replications of cross-checkpoint structure. This diagnostic does not identify the historical cause of the closeness; shared initialization/pretraining lineage is a leading interpretation, not established provenance.

## Files

- `matched_weight_similarity.csv`: all 216 pair/matrix measurements.
- `summary.json`: overall and per-role summaries plus exact input bindings.
- `run.log`: runtime stages.
"""
        (stage / "README.md").write_text(readme, encoding="utf-8")
        files = []
        for path in sorted(stage.iterdir()):
            if path.name == "artifact_manifest.json" or not path.is_file():
                continue
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        json_dump(
            stage / "artifact_manifest.json",
            {
                "schema_version": "global_context_weight_lineage_posthoc_manifest_v1",
                "manifest_self_excluded": True,
                "count": len(files),
                "artifacts": files,
            },
        )
        os.replace(stage, output_dir)
        log = logging.getLogger("weight_lineage_posthoc")
        print(f"status=PASS output_dir={output_dir}")
        print(f"summary={output_dir / 'summary.json'}")
        print(f"manifest={output_dir / 'artifact_manifest.json'}")
        print(
            "source_beans median_cosine="
            f"{sb['cosine']['median']:.9f} median_relative_l2={sb['relative_l2_to_left']['median']:.9f}"
        )
        print(
            "source_trocr median_cosine="
            f"{st['cosine']['median']:.9f} median_relative_l2={st['relative_l2_to_left']['median']:.9f}"
        )
    except Exception:
        logging.exception("status=FAIL")
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
