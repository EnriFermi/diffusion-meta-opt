from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from scripts.audit_variant_a_estimator_stability import sha256_file, stable_uint63
from scripts.audit_variant_a_hvp_batch_size import PROTOCOL_ID, TRAIN_COUNT, TRAIN_PAIRS, TRAIN_STATES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_hvp_batch_size_ablation_h2048"
)
DEFAULT_RUN_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
EXPECTED_GRID = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
EXPECTED_REPEATS = 12


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def expected_batch(
    *,
    batch_size: int,
    step: int,
    sample_key: int,
    pair_key: int,
) -> tuple[int | None, str]:
    if int(batch_size) >= TRAIN_COUNT:
        return None, "full"
    offset = (int(sample_key) * 1009 + int(step) * 9176 + int(pair_key) * 7919) % TRAIN_COUNT
    indices = (torch.arange(int(batch_size), dtype=torch.long) + int(offset)).remainder(TRAIN_COUNT)
    return int(offset), tensor_sha256(indices)


def symmetric_error(left: float, right: float) -> float:
    denominator = abs(float(left)) + abs(float(right))
    if denominator == 0.0:
        return 0.0
    return float(2.0 * abs(float(left) - float(right)) / denominator)


def gram_metrics(gram: np.ndarray, left: int, right: int) -> tuple[float, float, float]:
    left_norm2 = max(float(gram[left, left]), 0.0)
    right_norm2 = max(float(gram[right, right]), 0.0)
    dot = float(gram[left, right])
    cosine = dot / max(math.sqrt(left_norm2 * right_norm2), 1e-30)
    relative_error = math.sqrt(max(left_norm2 + right_norm2 - 2.0 * dot, 0.0)) / max(
        math.sqrt(right_norm2), 1e-30
    )
    norm_ratio = math.sqrt(left_norm2) / max(math.sqrt(right_norm2), 1e-30)
    return float(cosine), float(relative_error), float(norm_ratio)


def concatenated_gram_metrics(
    gram: np.ndarray,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
) -> tuple[float, float, float]:
    if len(left_indices) != len(right_indices) or not left_indices:
        raise ValueError("left/right index lists must have equal nonzero length")
    left_norm2 = sum(max(float(gram[index, index]), 0.0) for index in left_indices)
    right_norm2 = sum(max(float(gram[index, index]), 0.0) for index in right_indices)
    paired_dot = sum(float(gram[left, right]) for left, right in zip(left_indices, right_indices, strict=True))
    cosine = paired_dot / max(math.sqrt(left_norm2 * right_norm2), 1e-30)
    relative_error = math.sqrt(max(left_norm2 + right_norm2 - 2.0 * paired_dot, 0.0)) / max(
        math.sqrt(right_norm2), 1e-30
    )
    norm_ratio = math.sqrt(left_norm2) / max(math.sqrt(right_norm2), 1e-30)
    return float(cosine), float(relative_error), float(norm_ratio)


def top_mass_share(values: Sequence[float], fraction: float = 0.01) -> float:
    array = np.abs(np.asarray(values, dtype=np.float64))
    if array.size == 0 or float(array.sum()) <= 0.0:
        return float("nan")
    count = max(1, int(math.ceil(float(fraction) * int(array.size))))
    return float(np.sort(array)[-count:].sum() / array.sum())


def hvp_tail_summary(atomic: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    branch_rows = []
    shared = [*group_columns, "repeat", "state_position", "pair_position"]
    for row in atomic.to_dict(orient="records"):
        for branch in (1, 2):
            branch_rows.append(
                {
                    **{column: row[column] for column in shared},
                    "branch": branch - 1,
                    "hvp_norm": float(row[f"h{branch}_norm"]),
                }
            )
    branches = pd.DataFrame(branch_rows)
    rows = []
    for key, group in branches.groupby(list(group_columns), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        values = group["hvp_norm"].to_numpy(dtype=np.float64)
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "hvp_branch_count": int(len(values)),
                "hvp_norm_mean": float(values.mean()),
                "hvp_norm_median": float(np.median(values)),
                "hvp_norm_q90": float(np.quantile(values, 0.90)),
                "hvp_norm_q99": float(np.quantile(values, 0.99)),
                "hvp_norm_max": float(values.max()),
                "hvp_norm_top1pct_mass_share": top_mass_share(values),
            }
        )
    return pd.DataFrame(rows)


def artifact_hashes(output_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        "protocol.md",
        "resolved_config.json",
        "preflight.json",
        "active_parameters.csv",
        "sampled_states.csv",
        "sequential_memory_smoke.csv",
        "simultaneous_a_memory_smoke.csv",
        "atomic_pair_samples.csv",
        "atomic_paired_scalars.csv",
        "window_coverage.csv",
        "window_overlap.csv",
        "gradient_index.csv",
        "gradient_gram.npy",
        "module_gradient_grams.npz",
        "paired_update_metrics.csv",
        "module_paired_update_metrics.csv",
        "cross_repeat_gradient_metrics.csv",
        "reference_split_checks.csv",
        "batch_summary.csv",
        "task_tail_summary.csv",
        "endpoint_bootstrap.csv",
        "energy_weighted_paired_metrics.csv",
        "reference_repeat_energy.csv",
        "validity.json",
        "manifest.json",
        "run.log",
        "paired_full_convergence.png",
        "reproducibility_and_tails.png",
        "window_coverage_and_cost.png",
    ]
    result = {}
    for name in names:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent posthoc validation of the HVP batch-size packet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()

    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    executor_validity = json.loads((output_dir / "validity.json").read_text(encoding="utf-8"))
    source_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    acceptance = json.loads((run_dir / "baseline_acceptance.json").read_text(encoding="utf-8"))
    atomic = pd.read_csv(output_dir / "atomic_pair_samples.csv")
    updates = pd.read_csv(output_dir / "paired_update_metrics.csv")
    summary = pd.read_csv(output_dir / "batch_summary.csv")
    coverage = pd.read_csv(output_dir / "window_coverage.csv")
    overlaps = pd.read_csv(output_dir / "window_overlap.csv")
    gradient_index = pd.read_csv(output_dir / "gradient_index.csv")
    gram = np.load(output_dir / "gradient_gram.npy", allow_pickle=False)
    module_archive = np.load(output_dir / "module_gradient_grams.npz", allow_pickle=False)
    module_grams = {name: module_archive[name] for name in module_archive.files}
    module_sum = np.zeros_like(gram)
    for value in module_grams.values():
        module_sum += value

    checks: dict[str, bool] = {}
    checks["executor_validity_passed"] = bool(executor_validity.get("passed"))
    checks["manifest_complete"] = manifest.get("status") == "complete"
    checks["exact_frozen_grid"] = list(resolved["batch_sizes"]) == EXPECTED_GRID
    checks["summary_grid_and_finiteness"] = (
        summary["requested_batch_size"].astype(int).tolist() == EXPECTED_GRID
        and bool(
            np.isfinite(
                summary[
                    [
                        "paired_gradient_cosine_median",
                        "paired_gradient_relative_error_median",
                        "paired_scalar_error_median",
                        "cross_repeat_cosine_median",
                        "atomic_scalar_q99",
                        "atomic_gradient_norm_q99",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )
    )
    checks["all_frozen_arms_completed"] = list(manifest["completed_batch_sizes"]) == EXPECTED_GRID
    checks["full_reference_used"] = int(manifest["reference_batch_size"]) == TRAIN_COUNT and bool(
        manifest["reference_is_full"]
    )
    checks["exact_train_budget"] = (
        int(resolved["states"]) == TRAIN_STATES
        and int(resolved["pairs"]) == TRAIN_PAIRS
        and int(resolved["repeats"]) == EXPECTED_REPEATS
    )
    checks["exact_train_step_keys"] = list(resolved["step_keys"]) == [
        10 * (repeat + 1) for repeat in range(EXPECTED_REPEATS)
    ]
    checks["source_config_semantics"] = (
        str(source_config["dtype"]) == "float32"
        and str(source_config["tiny_bigvae_output_mode"]) == "direct"
        and int(source_config["vae_hidden_dim"]) == 2048
        and int(source_config["latent_dim"]) == 512
        and int(source_config["train_subset"]) == TRAIN_COUNT
    )
    checks["acceptance_passed"] = bool(acceptance.get("passed"))
    checks["both_memory_smokes_complete"] = all(
        pd.read_csv(output_dir / filename)["requested_batch_size"].astype(int).tolist() == EXPECTED_GRID
        and bool(pd.read_csv(output_dir / filename)["success"].all())
        for filename in ("sequential_memory_smoke.csv", "simultaneous_a_memory_smoke.csv")
    )

    expected_atomic_count = len(EXPECTED_GRID) * EXPECTED_REPEATS * TRAIN_STATES * TRAIN_PAIRS
    checks["atomic_cardinality"] = len(atomic) == expected_atomic_count and not bool(
        atomic.duplicated(["requested_batch_size", "repeat", "state_position", "pair_position"]).any()
    )
    replay_failures = []
    for row in atomic.itertuples():
        for branch in (1, 2):
            pair_key = 2 * int(row.pair_position) + branch - 1
            expected_offset, expected_hash = expected_batch(
                batch_size=int(row.requested_batch_size),
                step=int(row.step_key),
                sample_key=int(row.source_weight_index),
                pair_key=pair_key,
            )
            observed_offset_raw = getattr(row, f"batch_{branch}_offset")
            observed_offset = None if pd.isna(observed_offset_raw) else int(observed_offset_raw)
            observed_hash = str(getattr(row, f"batch_{branch}_hash"))
            if observed_offset != expected_offset or observed_hash != expected_hash:
                replay_failures.append(
                    {
                        "batch_size": int(row.requested_batch_size),
                        "repeat": int(row.repeat),
                        "state": int(row.state_position),
                        "pair": int(row.pair_position),
                        "branch": branch - 1,
                        "expected_offset": expected_offset,
                        "observed_offset": observed_offset,
                        "expected_hash": expected_hash,
                        "observed_hash": observed_hash,
                    }
                )
            expected_probe_seed = stable_uint63(
                PROTOCOL_ID,
                int(resolved["seed"]),
                "probe",
                int(row.repeat),
                int(row.state_position),
                int(row.pair_position),
                branch - 1,
            )
            if int(getattr(row, f"probe_seed_{branch}")) != expected_probe_seed:
                replay_failures.append(
                    {
                        "batch_size": int(row.requested_batch_size),
                        "repeat": int(row.repeat),
                        "state": int(row.state_position),
                        "pair": int(row.pair_position),
                        "branch": branch - 1,
                        "probe_seed_mismatch": True,
                    }
                )
    checks["independent_batch_and_probe_replay"] = len(replay_failures) == 0

    coverage_checks = []
    overlap_checks = []
    for batch_size in EXPECTED_GRID:
        batch_atomic = atomic.loc[atomic["requested_batch_size"] == batch_size]
        for (repeat, state), group in batch_atomic.groupby(["repeat", "state_position"], sort=True):
            masks = []
            for row in group.sort_values("pair_position").itertuples():
                for branch in (1, 2):
                    raw_offset = getattr(row, f"batch_{branch}_offset")
                    if pd.isna(raw_offset):
                        mask = np.ones(TRAIN_COUNT, dtype=bool)
                    else:
                        indices = (int(raw_offset) + np.arange(int(batch_size), dtype=np.int64)) % TRAIN_COUNT
                        mask = np.zeros(TRAIN_COUNT, dtype=bool)
                        mask[indices] = True
                    masks.append(mask)
            multiplicity = np.stack(masks).sum(axis=0)
            observed = coverage.loc[
                (coverage["requested_batch_size"] == batch_size)
                & (coverage["repeat"] == repeat)
                & (coverage["state_position"] == state)
            ].iloc[0]
            coverage_checks.append(
                int(observed["union_count"]) == int((multiplicity > 0).sum())
                and int(observed["max_multiplicity"]) == int(multiplicity.max())
                and abs(float(observed["union_fraction"]) - float((multiplicity > 0).mean())) <= 1e-12
            )
            expected_pairs = []
            for left in range(len(masks)):
                for right in range(left + 1, len(masks)):
                    expected_pairs.append(int(np.logical_and(masks[left], masks[right]).sum()))
            observed_pairs = overlaps.loc[
                (overlaps["requested_batch_size"] == batch_size)
                & (overlaps["repeat"] == repeat)
                & (overlaps["state_position"] == state)
            ].sort_values(["left_pair", "left_branch", "right_pair", "right_branch"])
            overlap_checks.append(sorted(expected_pairs) == sorted(observed_pairs["overlap_count"].astype(int).tolist()))
    checks["independent_coverage_replay"] = bool(all(coverage_checks))
    checks["independent_overlap_replay"] = bool(all(overlap_checks))

    expected_gradient_count = len(EXPECTED_GRID) * EXPECTED_REPEATS
    checks["gradient_index_cardinality"] = (
        len(gradient_index) == expected_gradient_count
        and not bool(gradient_index.duplicated(["requested_batch_size", "repeat"]).any())
        and gradient_index["gradient_index"].astype(int).tolist() == list(range(expected_gradient_count))
    )
    checks["gram_shape_finite_symmetric"] = (
        gram.shape == (expected_gradient_count, expected_gradient_count)
        and bool(np.isfinite(gram).all())
        and bool(np.allclose(gram, gram.T, rtol=1e-12, atol=1e-6))
    )
    checks["module_grams_finite_symmetric"] = bool(
        module_grams
        and all(
            value.shape == gram.shape
            and np.isfinite(value).all()
            and np.allclose(value, value.T, rtol=1e-12, atol=1e-6)
            for value in module_grams.values()
        )
    )
    checks["module_grams_sum_to_full"] = bool(np.allclose(module_sum, gram, rtol=2e-12, atol=1e-5))
    checks["gram_norms_match_index"] = bool(
        np.allclose(
            np.sqrt(np.maximum(np.diag(gram), 0.0)),
            gradient_index["gradient_norm"].to_numpy(dtype=np.float64),
            rtol=2e-6,
            atol=2e-6,
        )
    )

    index_by_key = {
        (int(row.requested_batch_size), int(row.repeat)): int(row.gradient_index)
        for row in gradient_index.itertuples()
    }
    update_replay_errors = []
    for row in updates.itertuples():
        left = index_by_key[(int(row.requested_batch_size), int(row.repeat))]
        right = index_by_key[(TRAIN_COUNT, int(row.repeat))]
        cosine, relative_error, norm_ratio = gram_metrics(gram, left, right)
        expected_scalar_error = symmetric_error(float(row.a_scalar), float(row.reference_a_scalar))
        observed = np.asarray(
            [
                float(row.paired_gradient_cosine),
                float(row.paired_gradient_relative_error),
                float(row.paired_gradient_norm_ratio),
                float(row.paired_scalar_symmetric_error),
            ]
        )
        expected = np.asarray([cosine, relative_error, norm_ratio, expected_scalar_error])
        if not np.allclose(observed, expected, rtol=2e-12, atol=2e-12):
            update_replay_errors.append(
                {
                    "batch_size": int(row.requested_batch_size),
                    "repeat": int(row.repeat),
                    "observed": observed.tolist(),
                    "expected": expected.tolist(),
                }
            )
    checks["independent_update_metric_replay"] = len(update_replay_errors) == 0

    full_indices = [index_by_key[(TRAIN_COUNT, repeat)] for repeat in range(EXPECTED_REPEATS)]
    full_energy = np.asarray([max(float(gram[index, index]), 0.0) for index in full_indices])
    full_energy_total = max(float(full_energy.sum()), 1e-30)
    reference_energy = pd.DataFrame(
        {
            "repeat": list(range(EXPECTED_REPEATS)),
            "gradient_index": full_indices,
            "gradient_norm2": full_energy,
            "energy_share": full_energy / full_energy_total,
        }
    ).sort_values("energy_share", ascending=False)
    reference_energy["cumulative_energy_share"] = reference_energy["energy_share"].cumsum()
    reference_energy.to_csv(output_dir / "reference_repeat_energy.csv", index=False)
    top2_reference_energy_share = float(reference_energy["energy_share"].head(2).sum())

    energy_weighted_rows = []
    for batch_size in EXPECTED_GRID:
        batch_indices = [index_by_key[(batch_size, repeat)] for repeat in range(EXPECTED_REPEATS)]
        cosine, relative_error, norm_ratio = concatenated_gram_metrics(gram, batch_indices, full_indices)
        energy_weighted_rows.append(
            {
                "requested_batch_size": batch_size,
                "reference_batch_size": TRAIN_COUNT,
                "concatenated_gradient_cosine": cosine,
                "concatenated_gradient_relative_error": relative_error,
                "concatenated_gradient_norm_ratio": norm_ratio,
                "reference_top2_energy_share": top2_reference_energy_share,
            }
        )
    energy_weighted = pd.DataFrame(energy_weighted_rows)
    energy_weighted.to_csv(output_dir / "energy_weighted_paired_metrics.csv", index=False)
    checks["energy_weighted_metrics_finite"] = bool(
        np.isfinite(
            energy_weighted[
                [
                    "concatenated_gradient_cosine",
                    "concatenated_gradient_relative_error",
                    "concatenated_gradient_norm_ratio",
                    "reference_top2_energy_share",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )

    old_active = pd.read_csv(
        output_dir.parent / "a_estimator_stability_unfinetuned_h2048" / "active_parameters.csv"
    )
    new_active = pd.read_csv(output_dir / "active_parameters.csv")
    checks["active_parameter_manifest_matches_accepted_audit"] = bool(old_active.equals(new_active))
    checks["active_parameter_count_match"] = int(new_active["numel"].sum()) == 11685120

    print("[hvp_batch_review] stage=task_tensor_hashes", flush=True)
    from scripts.audit_variant_a_estimator_stability import _load_run

    loaded = _load_run(run_dir, device=torch.device("cpu"))
    task_hashes: dict[str, dict[str, Any]] = {}
    for task_name, task in sorted(loaded.task_tensors.items()):
        task_hashes[task_name] = {}
        for field in ("train_images", "train_labels", "test_images", "test_labels"):
            tensor = getattr(task, field)
            task_hashes[task_name][field] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
    checks["task_train_counts_match"] = bool(
        task_hashes and all(value["train_labels"]["shape"][0] == TRAIN_COUNT for value in task_hashes.values())
    )

    hvp_summary = hvp_tail_summary(atomic, ["requested_batch_size"])
    task_hvp_summary = hvp_tail_summary(atomic, ["requested_batch_size", "task_name"])
    hvp_summary.to_csv(output_dir / "hvp_norm_tail_summary.csv", index=False)
    task_hvp_summary.to_csv(output_dir / "task_hvp_norm_tail_summary.csv", index=False)
    checks["hvp_tail_cardinality"] = bool(
        (hvp_summary["hvp_branch_count"] == EXPECTED_REPEATS * TRAIN_STATES * TRAIN_PAIRS * 2).all()
        and int(task_hvp_summary["hvp_branch_count"].sum()) == 2 * expected_atomic_count
    )
    checks["hvp_tail_finite"] = bool(
        np.isfinite(
            hvp_summary[
                ["hvp_norm_mean", "hvp_norm_median", "hvp_norm_q90", "hvp_norm_q99", "hvp_norm_max"]
            ].to_numpy(dtype=float)
        ).all()
    )

    hashes = artifact_hashes(output_dir)
    hashes["source_config.json"] = {
        "path": str(run_dir / "config.json"),
        "sha256": sha256_file(run_dir / "config.json"),
        "size_bytes": int((run_dir / "config.json").stat().st_size),
    }
    hashes["baseline_acceptance.json"] = {
        "path": str(run_dir / "baseline_acceptance.json"),
        "sha256": sha256_file(run_dir / "baseline_acceptance.json"),
        "size_bytes": int((run_dir / "baseline_acceptance.json").stat().st_size),
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    posthoc = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "replay_failure_count": len(replay_failures),
        "replay_failures": replay_failures[:20],
        "update_metric_replay_error_count": len(update_replay_errors),
        "update_metric_replay_errors": update_replay_errors[:20],
        "source_config_sha256": sha256_file(run_dir / "config.json"),
        "acceptance_sha256": sha256_file(run_dir / "baseline_acceptance.json"),
        "task_tensor_hashes": task_hashes,
        "artifact_hash_manifest": str(output_dir / "artifact_hashes.json"),
        "hvp_norm_tail_summary": str(output_dir / "hvp_norm_tail_summary.csv"),
        "task_hvp_norm_tail_summary": str(output_dir / "task_hvp_norm_tail_summary.csv"),
        "energy_weighted_paired_metrics": str(output_dir / "energy_weighted_paired_metrics.csv"),
        "reference_repeat_energy": str(output_dir / "reference_repeat_energy.csv"),
        "reference_reproducible": bool(manifest["reference_reproducible"]),
    }
    (output_dir / "posthoc_validation.json").write_text(
        json.dumps(posthoc, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[hvp_batch_review] done {json.dumps(posthoc, sort_keys=True)}", flush=True)
    if not posthoc["passed"]:
        raise RuntimeError(f"posthoc validation failed: {json.dumps(posthoc, sort_keys=True)}")


if __name__ == "__main__":
    main()
