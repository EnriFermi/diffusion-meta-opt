from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = ["source_weight_index", "start_bank_position", "task_name", "tau"]
DEFAULT_OUTPUT_COLUMNS = [
    "discriminator_selection",
    "source_weight_index",
    "start_bank_position",
    "run",
    "step",
    "task_name",
    "tau",
    "optimizer",
    "source_lr",
    "weight_distribution",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "start_role",
    "selection",
    "vae_variant",
    "vae_geometry_reg_coeff",
    "delta_aulc",
    "delta_step0_test_loss",
    "delta_final_test_loss",
    "delta_reconstruction_rel_l2",
]


def _load_decoder_deltas(path: Path, method: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "method_analyzed" in rows.columns:
        rows = rows[rows["method_analyzed"].astype(str) == method].copy()
    missing = [col for col in KEY_COLS + ["delta_aulc", "delta_step0_test_loss"] if col not in rows.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows["source_weight_index"] = rows["source_weight_index"].astype(int)
    rows["start_bank_position"] = rows["start_bank_position"].astype(int)
    for col in ["delta_aulc", "delta_step0_test_loss", "delta_final_test_loss", "delta_reconstruction_rel_l2"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    if rows[KEY_COLS].duplicated().any():
        dupes = rows.loc[rows[KEY_COLS].duplicated(), KEY_COLS].head(5).to_dict(orient="records")
        raise ValueError(f"{path} has duplicate decoder paired keys, examples: {dupes}")
    if not np.isfinite(rows[["delta_aulc", "delta_step0_test_loss"]].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} has non-finite delta_aulc or delta_step0_test_loss values")
    return rows


def _load_eval_start_bank(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "source_weight_index" not in rows.columns:
        raise ValueError(f"{path} has no source_weight_index column")
    rows["source_weight_index"] = rows["source_weight_index"].astype(int)
    if "start_bank_position" in rows.columns:
        rows["start_bank_position"] = rows["start_bank_position"].astype(int)
    if "start_role" in rows.columns:
        rows = rows[rows["start_role"].astype(str) == "eval"].copy()
    if rows["source_weight_index"].duplicated().any():
        dupes = rows.loc[rows["source_weight_index"].duplicated(), "source_weight_index"].head(10).tolist()
        raise ValueError(f"{path} has duplicate eval source_weight_index values, examples: {dupes}")
    return rows


def _pick(
    pool: pd.DataFrame,
    *,
    selected_sources: set[int],
    label: str,
    count: int,
    sort_by: str,
    ascending: bool,
    positive_only: bool = False,
    negative_only: bool = False,
) -> pd.DataFrame:
    if count <= 0:
        return pool.iloc[0:0].copy()
    candidates = pool[~pool["source_weight_index"].astype(int).isin(selected_sources)].copy()
    if positive_only:
        candidates = candidates[pd.to_numeric(candidates[sort_by], errors="coerce") > 0.0].copy()
    if negative_only:
        candidates = candidates[pd.to_numeric(candidates[sort_by], errors="coerce") < 0.0].copy()
    candidates["_abs_delta_aulc"] = pd.to_numeric(candidates["delta_aulc"], errors="coerce").abs()
    sort_cols = [sort_by, "source_weight_index"] if sort_by != "_abs_delta_aulc" else ["_abs_delta_aulc", "source_weight_index"]
    ascending_values = [ascending, True]
    chosen = candidates.sort_values(sort_cols, ascending=ascending_values).head(int(count)).copy()
    if len(chosen) != int(count):
        raise ValueError(f"selection {label} requested {count} starts but found {len(chosen)}")
    chosen["discriminator_selection"] = label
    selected_sources.update(chosen["source_weight_index"].astype(int).tolist())
    return chosen.drop(columns=["_abs_delta_aulc"], errors="ignore")


def build_selection(
    *,
    paired_deltas_csv: Path,
    start_bank_csv: Path,
    output_csv: Path,
    method: str,
    a_worse_count: int,
    a_better_count: int,
    near_zero_count: int,
    step0_pos_count: int,
    step0_neg_count: int,
) -> pd.DataFrame:
    deltas = _load_decoder_deltas(paired_deltas_csv, method)
    bank = _load_eval_start_bank(start_bank_csv)
    merged = bank.merge(
        deltas[KEY_COLS + ["delta_aulc", "delta_step0_test_loss", "delta_final_test_loss", "delta_reconstruction_rel_l2"]],
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(deltas):
        raise ValueError(
            f"start bank / paired delta mismatch: merged {len(merged)} rows, deltas {len(deltas)} rows"
        )

    selected_sources: set[int] = set()
    parts = [
        _pick(
            merged,
            selected_sources=selected_sources,
            label="top_a_worse",
            count=a_worse_count,
            sort_by="delta_aulc",
            ascending=False,
        ),
        _pick(
            merged,
            selected_sources=selected_sources,
            label="top_a_better",
            count=a_better_count,
            sort_by="delta_aulc",
            ascending=True,
        ),
        _pick(
            merged,
            selected_sources=selected_sources,
            label="near_zero",
            count=near_zero_count,
            sort_by="_abs_delta_aulc",
            ascending=True,
        ),
        _pick(
            merged,
            selected_sources=selected_sources,
            label="step0_pos_outlier",
            count=step0_pos_count,
            sort_by="delta_step0_test_loss",
            ascending=False,
            positive_only=True,
        ),
        _pick(
            merged,
            selected_sources=selected_sources,
            label="step0_neg_outlier",
            count=step0_neg_count,
            sort_by="delta_step0_test_loss",
            ascending=True,
            negative_only=True,
        ),
    ]
    selected = pd.concat(parts, ignore_index=True)
    if selected["source_weight_index"].duplicated().any():
        raise AssertionError("selection produced duplicate source_weight_index values")

    columns = [col for col in DEFAULT_OUTPUT_COLUMNS if col in selected.columns]
    selected = selected[columns].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-deltas-csv", type=Path, required=True)
    parser.add_argument("--start-bank-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--method", default="decoder_latent")
    parser.add_argument("--a-worse-count", type=int, default=5)
    parser.add_argument("--a-better-count", type=int, default=5)
    parser.add_argument("--near-zero-count", type=int, default=5)
    parser.add_argument("--step0-pos-count", type=int, default=1)
    parser.add_argument("--step0-neg-count", type=int, default=0)
    args = parser.parse_args()
    selected = build_selection(
        paired_deltas_csv=args.paired_deltas_csv,
        start_bank_csv=args.start_bank_csv,
        output_csv=args.output_csv,
        method=str(args.method),
        a_worse_count=int(args.a_worse_count),
        a_better_count=int(args.a_better_count),
        near_zero_count=int(args.near_zero_count),
        step0_pos_count=int(args.step0_pos_count),
        step0_neg_count=int(args.step0_neg_count),
    )
    counts = selected["discriminator_selection"].value_counts().to_dict()
    sources = selected["source_weight_index"].astype(int).tolist()
    print(f"wrote {args.output_csv} rows={len(selected)} counts={counts} source_indices={sources}")


if __name__ == "__main__":
    main()
