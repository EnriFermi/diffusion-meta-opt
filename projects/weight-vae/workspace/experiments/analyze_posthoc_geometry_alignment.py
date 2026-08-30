#!/usr/bin/env python3
"""Post-hoc diagnostics for shared versus panel-specific Weight-AE latent geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/"
    "prospective_geometry_matched_replication_analysis_project_root_compatibility_repair_20260816"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "artifacts/crossmodal_united_structure/"
    "prospective_geometry_matched_replication_posthoc_geometry_alignment_20260816_v2"
)
EXPECTED_INPUT_MANIFEST_SHA256 = "e92d532261dd126324fbcecfbe90713763f74611ef2338772113fa13a81729aa"
PANELS = ("source_vit_b_flickr", "beans", "trocr_sroie")
PANEL_LABELS = {
    "source_vit_b_flickr": "source/Flickr",
    "beans": "Beans",
    "trocr_sroie": "TrOCR/SROIE",
}
ROLES = ("attn_query", "attn_key", "attn_value", "attn_output", "ffn_up", "ffn_down")
DEPTHS = tuple(range(12))
TILINGS = (1, 2)
PAIR_ORDER = (
    ("source_vit_b_flickr", "beans"),
    ("source_vit_b_flickr", "trocr_sroie"),
    ("beans", "trocr_sroie"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("posthoc_geometry_alignment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def verify_manifest(root: Path) -> dict[str, Any]:
    path = root / "artifact_manifest.json"
    observed_sha = sha256_file(path)
    if observed_sha != EXPECTED_INPUT_MANIFEST_SHA256:
        raise RuntimeError(
            f"formal analysis manifest SHA mismatch: {observed_sha} != {EXPECTED_INPUT_MANIFEST_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or payload.get("count") != len(rows):
        raise RuntimeError("formal analysis manifest list/count mismatch")
    declared = {str(row["path"]): row for row in rows}
    actual = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*")
        if file.is_file() and file != path
    }
    if set(declared) != actual:
        raise RuntimeError("formal analysis manifest is incomplete")
    for relative, row in declared.items():
        file = root / relative
        if file.is_symlink() or file.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"formal analysis artifact path/size mismatch: {relative}")
        if sha256_file(file) != str(row["sha256"]):
            raise RuntimeError(f"formal analysis artifact SHA mismatch: {relative}")
    return {
        "path": str(path.resolve()),
        "sha256": observed_sha,
        "artifacts": len(rows),
        "pass": True,
    }


def load_geometry(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    scores_path = root / "geometry_pca_scores.csv"
    npz_path = root / "geometry_pca_inputs_and_model.npz"
    frame = pd.read_csv(scores_path)
    required = {
        "panel_id",
        "tiling_index",
        "depth",
        "role",
        *(f"PC{index}" for index in range(1, 11)),
    }
    if not required.issubset(frame.columns) or len(frame) != 432:
        raise RuntimeError("geometry score table schema/row mismatch")
    archive = np.load(npz_path, allow_pickle=False)
    arrays = {key: archive[key] for key in archive.files}
    if arrays["standardized_features"].shape != (432, 2560):
        raise RuntimeError("standardized feature shape mismatch")
    pc10 = frame[[f"PC{index}" for index in range(1, 11)]].to_numpy(dtype=np.float64)
    pca_serialization_max_abs = float(np.max(np.abs(pc10 - arrays["pca_scores"])))
    if not np.allclose(pc10, arrays["pca_scores"], rtol=1e-13, atol=1e-13):
        raise RuntimeError(
            "CSV and NPZ PCA scores differ beyond decimal-serialization tolerance: "
            f"max_abs={pca_serialization_max_abs}"
        )
    if not np.isfinite(arrays["standardized_features"]).all():
        raise RuntimeError("standardized features contain non-finite values")
    key_columns = ["panel_id", "tiling_index", "depth", "role"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("geometry rows contain duplicate formal keys")
    expected = {
        (panel, tiling, depth, role)
        for panel in PANELS
        for tiling in TILINGS
        for depth in DEPTHS
        for role in ROLES
    }
    observed = set(frame[key_columns].itertuples(index=False, name=None))
    if observed != expected:
        raise RuntimeError("geometry rows do not form the expected balanced grid")
    return frame, arrays, {
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": sha256_file(scores_path),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "rows": len(frame),
        "features": arrays["standardized_features"].shape[1],
        "pca_csv_npz_exact_equal": bool(np.array_equal(pc10, arrays["pca_scores"])),
        "pca_csv_npz_max_abs": pca_serialization_max_abs,
        "pca_csv_npz_rtol": 1e-13,
        "pca_csv_npz_atol": 1e-13,
        "pass": True,
    }


def balanced_tensor(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    lookup = {
        (str(row.panel_id), int(row.tiling_index), int(row.depth), str(row.role)): index
        for index, row in enumerate(frame.itertuples(index=False))
    }
    tensor = np.empty((len(PANELS), len(TILINGS), len(DEPTHS), len(ROLES), values.shape[1]))
    for panel_index, panel in enumerate(PANELS):
        for tiling_index, tiling in enumerate(TILINGS):
            for depth_index, depth in enumerate(DEPTHS):
                for role_index, role in enumerate(ROLES):
                    tensor[panel_index, tiling_index, depth_index, role_index] = values[
                        lookup[(panel, tiling, depth, role)]
                    ]
    return tensor


def variance_decomposition(tensor: np.ndarray, space: str) -> dict[str, Any]:
    panels, tilings, depths, roles, _ = tensor.shape
    cells = depths * roles
    flat = tensor.reshape(panels, tilings, cells, -1)
    grand = flat.mean(axis=(0, 1, 2))
    cell_mean = flat.mean(axis=(0, 1))
    panel_mean = flat.mean(axis=(1, 2))
    panel_cell_mean = flat.mean(axis=1)
    ss_total = float(np.square(flat - grand).sum())
    ss_cell = float(panels * tilings * np.square(cell_mean - grand).sum())
    ss_panel = float(tilings * cells * np.square(panel_mean - grand).sum())
    interaction = panel_cell_mean - cell_mean[None, :, :] - panel_mean[:, None, :] + grand
    ss_interaction = float(tilings * np.square(interaction).sum())
    ss_tiling = float(np.square(flat - panel_cell_mean[:, None, :, :]).sum())
    components = {
        "shared_role_x_depth_cell": ss_cell,
        "panel_main_effect": ss_panel,
        "panel_x_cell_interaction": ss_interaction,
        "tiling_residual": ss_tiling,
    }
    relative_error = abs(sum(components.values()) - ss_total) / max(ss_total, 1e-300)
    if relative_error > 1e-10:
        raise RuntimeError(f"balanced variance decomposition failed in {space}: {relative_error}")
    return {
        "space": space,
        "ss_total": ss_total,
        "components": components,
        "fractions": {key: value / ss_total for key, value in components.items()},
        "relative_sum_error": relative_error,
    }


def geometry_diagnostics(
    frame: pd.DataFrame,
    standardized: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tensor = balanced_tensor(frame, standardized)
    panel_cell = tensor.mean(axis=1).reshape(len(PANELS), 72, -1)
    tiling_rms = np.linalg.norm(tensor[:, 0] - tensor[:, 1], axis=-1) / math.sqrt(
        tensor.shape[-1]
    )
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for left, right in PAIR_ORDER:
        left_index = PANELS.index(left)
        right_index = PANELS.index(right)
        left_cells = panel_cell[left_index]
        right_cells = panel_cell[right_index]
        all_cross = np.linalg.norm(
            left_cells[:, None, :] - right_cells[None, :, :], axis=-1
        ) / math.sqrt(tensor.shape[-1])
        pair_rows: list[dict[str, Any]] = []
        for depth_index, depth in enumerate(DEPTHS):
            for role_index, role in enumerate(ROLES):
                cell_index = depth_index * len(ROLES) + role_index
                matched = float(all_cross[cell_index, cell_index])
                noise_left = float(tiling_rms[left_index, depth_index, role_index])
                noise_right = float(tiling_rms[right_index, depth_index, role_index])
                noise = 0.5 * (noise_left + noise_right)
                mismatched = np.delete(all_cross[cell_index], cell_index)
                record = {
                    "panel_left": left,
                    "panel_right": right,
                    "depth": depth,
                    "role": role,
                    "matched_rms_standardized": matched,
                    "tiling_rms_left": noise_left,
                    "tiling_rms_right": noise_right,
                    "mean_tiling_rms": noise,
                    "matched_over_tiling": matched / max(noise, 1e-12),
                    "median_mismatched_rms": float(np.median(mismatched)),
                    "matched_over_median_mismatched": matched
                    / max(float(np.median(mismatched)), 1e-12),
                    "matched_percentile_among_72": float(
                        100.0 * np.mean(all_cross[cell_index] <= matched)
                    ),
                }
                rows.append(record)
                pair_rows.append(record)
        pair_frame = pd.DataFrame(pair_rows)
        summary_rows.append(
            {
                "panel_left": left,
                "panel_right": right,
                "median_matched_rms_standardized": float(
                    pair_frame["matched_rms_standardized"].median()
                ),
                "median_mean_tiling_rms": float(pair_frame["mean_tiling_rms"].median()),
                "median_matched_over_tiling": float(
                    pair_frame["matched_over_tiling"].median()
                ),
                "median_matched_over_median_mismatched": float(
                    pair_frame["matched_over_median_mismatched"].median()
                ),
                "median_matched_percentile_among_72": float(
                    pair_frame["matched_percentile_among_72"].median()
                ),
            }
        )
    matched = pd.DataFrame(rows)

    rsa_rows: list[dict[str, Any]] = []
    condensed = {
        panel: pdist(panel_cell[index], metric="euclidean")
        for index, panel in enumerate(PANELS)
    }
    for left in PANELS:
        for right in PANELS:
            correlation, pvalue = spearmanr(condensed[left], condensed[right])
            rsa_rows.append(
                {
                    "panel_left": left,
                    "panel_right": right,
                    "spearman_rho": float(correlation),
                    "pvalue_descriptive_only": float(pvalue),
                    "pairwise_distances": len(condensed[left]),
                }
            )
    return matched, pd.DataFrame(summary_rows), pd.DataFrame(rsa_rows)


def render_variance_plot(rows: pd.DataFrame, output: Path) -> None:
    order = [
        "shared_role_x_depth_cell",
        "panel_main_effect",
        "panel_x_cell_interaction",
        "tiling_residual",
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b7b7b7"]
    fig, axis = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(rows))
    for component, color in zip(order, colors, strict=True):
        values = rows[component].to_numpy(dtype=float)
        axis.bar(rows["space"], values, bottom=bottom, label=component, color=color)
        bottom += values
    axis.set_ylim(0, 1)
    axis.set_ylabel("fraction of total sum of squares")
    axis.set_title("Balanced latent-geometry variance decomposition (post-hoc)")
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_distance_heatmaps(matched: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    matrices = []
    for left, right in PAIR_ORDER:
        subset = matched[(matched.panel_left == left) & (matched.panel_right == right)]
        matrix = (
            subset.pivot(index="role", columns="depth", values="matched_over_tiling")
            .reindex(index=ROLES, columns=DEPTHS)
            .to_numpy(dtype=float)
        )
        matrices.append(np.log10(np.maximum(matrix, 1e-12)))
    limit = max(abs(float(np.nanmin(matrices))), abs(float(np.nanmax(matrices))))
    for axis, matrix, (left, right) in zip(axes, matrices, PAIR_ORDER, strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"{PANEL_LABELS[left]} vs {PANEL_LABELS[right]}")
        axis.set_xticks(range(len(DEPTHS)), DEPTHS)
        axis.set_xlabel("depth")
        axis.set_yticks(range(len(ROLES)), ROLES)
    axes[0].set_ylabel("role")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("log10(matched panel distance / mean tiling distance)")
    fig.suptitle("Where panel shifts exceed tiling instability (post-hoc)", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.84, wspace=0.12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_rsa(rsa: pd.DataFrame, output: Path) -> None:
    matrix = (
        rsa.pivot(index="panel_left", columns="panel_right", values="spearman_rho")
        .reindex(index=PANELS, columns=PANELS)
        .to_numpy(dtype=float)
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    labels = [PANEL_LABELS[panel] for panel in PANELS]
    axis.set_xticks(range(3), labels, rotation=25, ha="right")
    axis.set_yticks(range(3), labels)
    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 0.65 else "black",
            )
    axis.set_title("RSA of the 72-cell role×depth geometry (post-hoc)")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Spearman correlation of pairwise distances")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def artifact_manifest(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "artifact_manifest.json":
            continue
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "artifacts": rows,
        "count": len(rows),
        "manifest_self_excluded": True,
        "schema_version": "posthoc_geometry_alignment_manifest_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"post-hoc output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    logger = setup_logging(output)
    started = time.monotonic()
    logger.info(
        "resolved_config input=%s output=%s device=cpu dtype=FP64 seed=none "
        "cache_mode=sealed_formal_geometry posthoc=true verbose=true",
        input_dir,
        output,
    )

    logger.info("stage=formal_artifact_manifest_audit")
    input_audit = verify_manifest(input_dir)
    frame, arrays, geometry_audit = load_geometry(input_dir)
    write_json(output / "input_audit.json", {"formal_manifest": input_audit, "geometry": geometry_audit})

    logger.info("stage=balanced_variance_decomposition spaces=full,PC10,PC2")
    spaces = {
        "standardized_full_2560D": arrays["standardized_features"],
        "PCA_10D": arrays["pca_scores"],
        "PCA_PC1_PC2": arrays["pca_scores"][:, :2],
    }
    decompositions = [
        variance_decomposition(balanced_tensor(frame, values), name)
        for name, values in spaces.items()
    ]
    write_json(output / "variance_decomposition.json", {"decompositions": decompositions})
    variance_rows = pd.DataFrame(
        [{"space": row["space"], **row["fractions"]} for row in decompositions]
    )
    variance_rows.to_csv(output / "variance_decomposition.csv", index=False)
    render_variance_plot(variance_rows, output / "variance_decomposition.png")

    logger.info("stage=matched_panel_distances_and_tiling_noise cells_per_panel=72")
    matched, matched_summary, rsa = geometry_diagnostics(frame, arrays["standardized_features"])
    matched.to_csv(output / "matched_panel_distances.csv", index=False)
    matched_summary.to_csv(output / "matched_panel_distance_summary.csv", index=False)
    render_distance_heatmaps(matched, output / "matched_panel_distance_heatmaps.png")

    logger.info("stage=representational_similarity pairwise_cell_distances=2556_per_panel")
    rsa.to_csv(output / "panel_geometry_rsa.csv", index=False)
    render_rsa(rsa, output / "panel_geometry_rsa.png")

    summary = {
        "schema_version": "posthoc_geometry_alignment_v1",
        "status": "COMPLETE",
        "scope": "exploratory post-hoc; non-gating; three panels with dataset/model/domain confounded",
        "formal_manifest_sha256": input_audit["sha256"],
        "variance_decomposition_full": decompositions[0]["fractions"],
        "matched_panel_distance_summary": matched_summary.to_dict("records"),
        "rsa": [
            rsa[(rsa.panel_left == left) & (rsa.panel_right == right)].iloc[0].to_dict()
            for left, right in PAIR_ORDER
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Post-hoc latent geometry alignment diagnostics\n\n"
        "These diagnostics are exploratory and do not change the frozen gate outcome. "
        "Dataset, model, and panel/domain labels are perfectly confounded because there is one checkpoint per panel.\n\n"
        "The balanced variance decomposition separates shared role×depth cell identity, a panel main effect, "
        "panel-specific deformation, and two-tiling residual noise. Matched-distance heatmaps compare cross-panel "
        "shifts with tiling instability. RSA compares the relational 72-cell geometry across panels.\n\n"
        "See `summary.json`, the three CSV tables, and the three PNG plots for exact evidence.\n",
        encoding="utf-8",
    )
    logger.info(
        "stage=complete elapsed=%.2fs summary=%s plots=%s",
        time.monotonic() - started,
        output / "summary.json",
        [
            str(output / "variance_decomposition.png"),
            str(output / "matched_panel_distance_heatmaps.png"),
            str(output / "panel_geometry_rsa.png"),
        ],
    )
    logger.info("stage=artifact_manifest final_write=true")
    for handler in logger.handlers:
        handler.flush()
    write_json(output / "artifact_manifest.json", artifact_manifest(output))


if __name__ == "__main__":
    main()
