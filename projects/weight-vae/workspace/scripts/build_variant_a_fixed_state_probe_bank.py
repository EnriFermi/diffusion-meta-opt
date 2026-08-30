from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a_fixed_state_probe_variance_h2048_v1"
SEED = 20260714
RUN_DIR = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing/"
    "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
OUTPUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "a_fixed_state_probe_variance_h2048"
)
PARENT_ATOMIC_PATH = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "a_hvp_batch_size_ablation_h2048/atomic_pair_samples.csv"
)
TASK_ORDER = ("fashion_mnist", "mnist")
STEP_STRATA = {
    0: (0, 500),
    1: (550, 1000),
    2: (1050, 1500),
    3: (1550, 2000),
}


def stable_uint63(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & (
        (1 << 63) - 1
    )


def hash_text(value: int) -> str:
    return f"u63_{value}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step_stratum(step: int) -> int:
    for stratum, (lo, hi) in STEP_STRATA.items():
        if lo <= step <= hi:
            return stratum
    raise ValueError(f"step {step} is outside the frozen strata")


def build_sentinel_parent(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    atomic = pd.read_csv(PARENT_ATOMIC_PATH)
    full = atomic.loc[atomic["requested_batch_size"].eq(16384)].copy()
    grouped = (
        full.groupby(
            ["task_name", "source_weight_index", "repeat", "state_position"],
            as_index=False,
            sort=False,
        )
        .agg(
            prior_grad_rms=(
                "gradient_norm",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
            atomic_count=("gradient_norm", "size"),
        )
        .merge(
            records[["source_weight_index", "run", "step", "tau"]],
            on="source_weight_index",
            how="left",
            validate="one_to_one",
        )
    )
    if len(grouped) != 24 or not grouped["atomic_count"].eq(4).all():
        raise RuntimeError("expected 24 parent states with four full-CE atomic gradients each")

    selected_rows: list[dict[str, object]] = []
    parent_parts: list[pd.DataFrame] = []
    for task_name in TASK_ORDER:
        task = grouped.loc[grouped["task_name"].eq(task_name)].copy()
        task = task.sort_values(
            ["prior_grad_rms", "source_weight_index"], kind="stable"
        ).reset_index(drop=True)
        task["sorted_rank"] = np.arange(1, len(task) + 1)
        task["median_rank_target"] = (len(task) + 1) / 2
        task["median_rank_distance"] = (
            task["sorted_rank"] - task["median_rank_target"]
        ).abs()

        low_indices: list[int] = []
        used_runs: set[int] = set()
        for idx, row in task.iterrows():
            run = int(row["run"])
            if run not in used_runs:
                low_indices.append(idx)
                used_runs.add(run)
            if len(low_indices) == 2:
                break

        high_indices: list[int] = []
        for idx, row in task.iloc[::-1].iterrows():
            run = int(row["run"])
            if run not in used_runs:
                high_indices.append(idx)
                used_runs.add(run)
            if len(high_indices) == 2:
                break
        high_indices.sort()

        middle_candidates = task.loc[~task["run"].isin(used_runs)].sort_values(
            ["median_rank_distance", "sorted_rank", "source_weight_index"],
            kind="stable",
        )
        middle_indices: list[int] = []
        middle_runs: set[int] = set()
        for idx, row in middle_candidates.iterrows():
            run = int(row["run"])
            if run not in middle_runs:
                middle_indices.append(idx)
                middle_runs.add(run)
            if len(middle_indices) == 2:
                break
        middle_indices.sort()

        chosen = {
            "low": low_indices,
            "middle": middle_indices,
            "high": high_indices,
        }
        task["low_candidate"] = task.index.isin(low_indices)
        task["high_candidate"] = task.index.isin(high_indices)
        task["middle_eligible_after_extreme_runs"] = ~task["run"].isin(used_runs)
        task["selected"] = False
        task["selected_stratum"] = ""
        task["selection_rank"] = pd.Series([pd.NA] * len(task), dtype="Int64")
        for prior_stratum, indices in chosen.items():
            for rank, idx in enumerate(indices, start=1):
                task.loc[idx, "selected"] = True
                task.loc[idx, "selected_stratum"] = prior_stratum
                task.loc[idx, "selection_rank"] = rank
                row = task.loc[idx]
                selected_rows.append(
                    {
                        "panel": "sentinel",
                        "task_name": task_name,
                        "source_weight_index": int(row["source_weight_index"]),
                        "run": int(row["run"]),
                        "step": int(row["step"]),
                        "step_stratum": step_stratum(int(row["step"])),
                        "tau": float(row["tau"]),
                        "selection_policy": "prior_full_p4_rms_rank",
                        "selection_rank": rank,
                        "run_selection_hash": "",
                        "snapshot_selection_hash": "",
                        "prior_grad_rms": float(row["prior_grad_rms"]),
                        "prior_stratum": prior_stratum,
                    }
                )
        parent_parts.append(task)

    parent = pd.concat(parent_parts, ignore_index=True)
    parent["selection_rule"] = (
        "low/high: two extreme RMS states on distinct runs; middle: two states "
        "closest to the task median rank after excluding low/high runs, lower rank first"
    )
    parent = parent[
        [
            "task_name",
            "source_weight_index",
            "run",
            "step",
            "tau",
            "repeat",
            "state_position",
            "atomic_count",
            "prior_grad_rms",
            "sorted_rank",
            "median_rank_target",
            "median_rank_distance",
            "low_candidate",
            "high_candidate",
            "middle_eligible_after_extreme_runs",
            "selected",
            "selected_stratum",
            "selection_rank",
            "selection_rule",
        ]
    ]
    sentinels = pd.DataFrame(selected_rows)
    return parent, sentinels


def build_primary(
    records: pd.DataFrame,
    train_indices: set[int],
    excluded_indices: set[int],
    sentinel_runs: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = records.loc[
        records["source_weight_index"].isin(train_indices)
        & ~records["source_weight_index"].isin(excluded_indices)
        & ~records["run"].isin(sentinel_runs)
    ].copy()
    candidates["step_stratum"] = candidates["step"].map(step_stratum)

    used_runs: set[int] = set()
    manifest_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    for task_name in TASK_ORDER:
        for stratum in STEP_STRATA:
            cell = candidates.loc[
                candidates["task_name"].eq(task_name)
                & candidates["step_stratum"].eq(stratum)
            ]
            run_rows: list[tuple[int, int, pd.DataFrame]] = []
            for run, snapshots in cell.groupby("run", sort=False):
                run_hash = stable_uint63(
                    PROTOCOL_ID, SEED, "primary_run", task_name, stratum, int(run)
                )
                run_rows.append((run_hash, int(run), snapshots))
            run_rows.sort(key=lambda item: (item[0], item[1]))

            available_rank = 0
            selected_count = 0
            for run_hash_rank, (run_hash, run, snapshots) in enumerate(run_rows, start=1):
                available = run not in used_runs
                if available:
                    available_rank += 1
                snapshot_candidates = []
                for source_index in snapshots["source_weight_index"]:
                    snapshot_hash = stable_uint63(
                        PROTOCOL_ID,
                        SEED,
                        "primary_snapshot",
                        task_name,
                        stratum,
                        run,
                        int(source_index),
                    )
                    snapshot_candidates.append((snapshot_hash, int(source_index)))
                snapshot_hash, source_index = min(snapshot_candidates)
                selected = available and selected_count < 4
                selection_rank: int | None = None
                if selected:
                    selected_count += 1
                    selection_rank = selected_count
                    used_runs.add(run)
                    record = records.loc[
                        records["source_weight_index"].eq(source_index)
                    ].iloc[0]
                    primary_rows.append(
                        {
                            "panel": "primary",
                            "task_name": task_name,
                            "source_weight_index": source_index,
                            "run": run,
                            "step": int(record["step"]),
                            "step_stratum": stratum,
                            "tau": float(record["tau"]),
                            "selection_policy": "task_step_stratified_unique_run_hash",
                            "selection_rank": selection_rank,
                            "run_selection_hash": hash_text(run_hash),
                            "snapshot_selection_hash": hash_text(snapshot_hash),
                            "prior_grad_rms": np.nan,
                            "prior_stratum": "",
                        }
                    )
                manifest_rows.append(
                    {
                        "task_name": task_name,
                        "step_stratum": stratum,
                        "run": run,
                        "run_selection_hash": hash_text(run_hash),
                        "run_hash_rank": run_hash_rank,
                        "available_when_visited": available,
                        "available_rank": available_rank if available else pd.NA,
                        "candidate_snapshot_count": len(snapshot_candidates),
                        "best_source_weight_index": source_index,
                        "best_snapshot_selection_hash": hash_text(snapshot_hash),
                        "selected": selected,
                        "selection_rank": selection_rank if selected else pd.NA,
                    }
                )
            if selected_count != 4:
                raise RuntimeError(
                    f"expected four primary runs in {(task_name, stratum)}, got {selected_count}"
                )

    primary = pd.DataFrame(primary_rows)
    eligible = pd.DataFrame(manifest_rows)
    eligible["available_rank"] = eligible["available_rank"].astype("Int64")
    eligible["selection_rank"] = eligible["selection_rank"].astype("Int64")
    return eligible, primary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(RUN_DIR / "weight_pool_records.csv").reset_index(
        names="source_weight_index"
    )
    checkpoint = torch.load(
        RUN_DIR / "vae_checkpoint.pt", map_location="cpu", weights_only=False
    )
    train_indices = {int(value) for value in checkpoint["train_indices"].tolist()}
    excluded = pd.read_csv(OUTPUT_DIR / "excluded_primary_sources.csv")
    excluded_indices = {int(value) for value in excluded["source_weight_index"]}

    sentinel_parent, sentinels = build_sentinel_parent(records)
    sentinel_runs = {int(value) for value in sentinels["run"]}
    eligible, primary = build_primary(
        records, train_indices, excluded_indices, sentinel_runs
    )
    bank = pd.concat([primary, sentinels], ignore_index=True)
    bank.insert(0, "state_position", np.arange(len(bank)))

    if len(primary) != 32 or primary["run"].nunique() != 32:
        raise RuntimeError("primary panel must contain 32 unique trajectories")
    if len(sentinels) != 12 or sentinels["run"].nunique() != 12:
        raise RuntimeError("sentinel panel must contain 12 unique trajectories")
    if set(primary["source_weight_index"]) & excluded_indices:
        raise RuntimeError("primary panel overlaps the frozen exclusion manifest")

    eligible.to_csv(OUTPUT_DIR / "eligible_runs.csv", index=False)
    sentinel_parent.to_csv(OUTPUT_DIR / "sentinel_parent_states.csv", index=False)
    bank.to_csv(OUTPUT_DIR / "state_bank.csv", index=False, float_format="%.15g")
    for name in ("eligible_runs.csv", "sentinel_parent_states.csv", "state_bank.csv"):
        path = OUTPUT_DIR / name
        print(f"{name}: rows={sum(1 for _ in path.open()) - 1} sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
