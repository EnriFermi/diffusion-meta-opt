#!/usr/bin/env python3
"""Build readable all-run tables from the TinyStories long measurement bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


REPORT_TYPES = (
    "external_generation_eval",
    "online_quality_curve",
    "validation_curve",
    "checkpoint_metadata",
    "sequence_latency_benchmark",
    "internal_checkpoint_eval",
    "sanity_eval_duplicate",
    "generation_benchmark",
    "model_inventory",
)

INDEX_COLUMNS = [
    "measurement_id",
    "measurement_type",
    "protocol_id",
    "comparable_group",
    "model_id",
    "model_family",
    "model_variant",
    "model_status",
    "run_id",
    "run_complete",
    "checkpoint_path",
    "checkpoint_step",
    "training_step",
    "epoch",
    "evaluation_label",
    "sampler",
    "block_size",
    "point",
    "discretize",
    "nfe",
    "forwards_per_sequence",
    "n_samples",
    "seed",
    "source_file",
]


def _join_unique(series: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in series.dropna()}))


def _first(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else None


def _wide(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    metadata = frame[INDEX_COLUMNS].drop_duplicates("measurement_id")
    values = frame.pivot_table(
        index="measurement_id",
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    return metadata.merge(values, on="measurement_id", how="left")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    columns = [column for column in columns if column in frame.columns]
    if frame.empty or not columns:
        return "_No canonical valid rows._\n"

    def render(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.7g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[config] "
        + json.dumps(
            {
                "input": str(args.input.resolve()),
                "output_dir": str(output_dir),
                "device": "cpu",
                "dtype": "CSV numeric float64",
                "seed": None,
                "cache_mode": "none",
            }
        ),
        flush=True,
    )

    print("[stage=load] reading long CSV", flush=True)
    frame = pd.read_csv(args.input, low_memory=False)
    print(
        f"[stage=load_done] rows={len(frame)} columns={len(frame.columns)} "
        f"runs={frame.run_id.nunique()} metrics={frame.metric_name.nunique()} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )

    print("[stage=validity] checking schema and row identities", flush=True)
    if set(frame.dataset.dropna().unique()) != {"tinystories"}:
        raise RuntimeError("bundle contains a non-TinyStories dataset")
    if frame.metric_row_id.duplicated().any():
        raise RuntimeError("metric_row_id is not unique")
    invalid_nan_rows = frame.metric_value.isna() & ~frame.valid_value
    valid_nan_rows = frame.metric_value.isna() & frame.valid_value
    if valid_nan_rows.any():
        raise RuntimeError("a row marked valid contains NaN metric_value")
    print(
        f"[validity] invalid_nan_rows={int(invalid_nan_rows.sum())} "
        "(retained in source, excluded from valid summaries)",
        flush=True,
    )

    print("[stage=inventory] aggregating runs", flush=True)
    inventory_fields = [
        "model_id",
        "model_family",
        "model_variant",
        "model_status",
        "run_complete",
        "tokenizer",
        "vocab_size",
        "sequence_length",
        "parameter_count",
        "hidden_size",
        "num_layers",
        "num_heads",
        "architecture",
        "target_training_steps",
    ]
    inventory_rows = []
    for run_id, group in frame.groupby("run_id", dropna=False, sort=True):
        row = {"run_id": run_id}
        row.update({field: _first(group[field]) for field in inventory_fields})
        row.update(
            {
                "checkpoint_paths": _join_unique(group.checkpoint_path),
                "measurement_types": _join_unique(group.measurement_type),
                "row_count": len(group),
                "measurement_count": group.measurement_id.nunique(),
                "metric_count": group.metric_name.nunique(),
                "min_training_step": group.training_step.min(),
                "max_training_step": group.training_step.max(),
                "canonical_rows": int(group.canonical_for_plot.sum()),
                "valid_rows": int(group.valid_value.sum()),
                "duplicate_rows": int(group.duplicate_of.notna().sum()),
            }
        )
        inventory_rows.append(row)
    inventory = pd.DataFrame(inventory_rows)
    inventory_path = output_dir / "run_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    print("[stage=catalog] aggregating every run/type/metric", flush=True)
    valid = frame[frame.valid_value].copy()
    order_step = valid.training_step.fillna(valid.checkpoint_step).fillna(-1)
    order_time = valid.event_wall_time_unix.fillna(-1)
    valid = valid.assign(_order_step=order_step, _order_time=order_time).sort_values(
        ["_order_step", "_order_time", "source_row"], kind="stable"
    )
    group_columns = [
        "run_id",
        "model_id",
        "model_family",
        "model_variant",
        "model_status",
        "measurement_type",
        "metric_name",
        "metric_unit",
    ]
    catalog = (
        valid.groupby(group_columns, dropna=False, sort=True)
        .agg(
            row_count=("metric_value", "size"),
            measurement_count=("measurement_id", "nunique"),
            canonical_rows=("canonical_for_plot", "sum"),
            duplicate_rows=("duplicate_of", lambda values: int(values.notna().sum())),
            min_value=("metric_value", "min"),
            max_value=("metric_value", "max"),
            mean_value=("metric_value", "mean"),
            latest_value=("metric_value", "last"),
            min_training_step=("training_step", "min"),
            max_training_step=("training_step", "max"),
            source_files=("source_file", _join_unique),
        )
        .reset_index()
    )
    catalog_path = output_dir / "all_runs_all_metrics_summary.csv"
    catalog.to_csv(catalog_path, index=False)

    print("[stage=pivot] writing canonical evaluation tables", flush=True)
    canonical = frame[
        frame.canonical_for_plot & frame.valid_value & frame.duplicate_of.isna()
    ].copy()
    report = canonical[canonical.measurement_type.isin(REPORT_TYPES)]
    all_wide = _wide(report)
    all_wide_path = output_dir / "canonical_evaluation_metrics_wide.csv"
    all_wide.to_csv(all_wide_path, index=False)
    table_manifest = {}
    for measurement_type in REPORT_TYPES:
        table = _wide(report[report.measurement_type == measurement_type])
        path = output_dir / f"{measurement_type}_wide.csv"
        table.to_csv(path, index=False)
        table_manifest[measurement_type] = {
            "rows": len(table),
            "metric_columns": (
                sorted(set(table.columns) - set(INDEX_COLUMNS)) if len(table) else []
            ),
            "path": str(path),
        }
        print(
            f"[table] type={measurement_type} rows={len(table)} path={path}",
            flush=True,
        )

    print("[stage=review] checking coverage", flush=True)
    if set(inventory.run_id.dropna()) != set(frame.run_id.dropna().unique()):
        raise RuntimeError("run inventory coverage mismatch")
    catalog_pairs = set(zip(catalog.run_id, catalog.metric_name))
    source_pairs = set(zip(valid.run_id, valid.metric_name))
    if catalog_pairs != source_pairs:
        raise RuntimeError("metric catalog coverage mismatch")

    external = _wide(
        report[report.measurement_type == "external_generation_eval"]
    )
    best_rows = []
    if "mauve" in external:
        candidates = external.dropna(subset=["mauve"])
        for _, group in candidates.groupby(
            ["model_id", "model_variant"], dropna=False, sort=True
        ):
            best_rows.append(group.loc[group.mauve.idxmax()])
    best = pd.DataFrame(best_rows)
    best_columns = [
        column
        for column in [
            "model_family",
            "model_variant",
            "model_status",
            "run_id",
            "evaluation_label",
            "sampler",
            "point",
            "nfe",
            "n_samples",
            "mauve",
            "gen_ppl",
            "token_entropy",
            "seq_rep_2",
            "distinct_2",
            "js_1gram",
            "js_2gram",
            "sampling_seconds",
        ]
        if column in best.columns
    ]
    best = best[best_columns] if len(best) else best
    best_path = output_dir / "best_mauve_point_per_model_variant.csv"
    best.to_csv(best_path, index=False)

    print("[stage=report] writing complete Markdown view", flush=True)
    online = _wide(report[report.measurement_type == "online_quality_curve"])
    validation = _wide(report[report.measurement_type == "validation_curve"])
    latency = _wide(report[report.measurement_type == "sequence_latency_benchmark"])
    checkpoint = _wide(report[report.measurement_type == "checkpoint_metadata"])
    internal = _wide(report[report.measurement_type == "internal_checkpoint_eval"])
    generation = _wide(report[report.measurement_type == "generation_benchmark"])

    report_parts = [
        "# TinyStories: all runs and all metrics\n",
        "This report is generated from `tinystories_all_measurements_long.csv`. "
        "Canonical tables keep only `canonical_for_plot=True`, "
        "`valid_value=True`, and rows without `duplicate_of`. The final catalog "
        "still covers every valid metric name for every run, including training "
        "and TensorBoard time series via count/min/max/mean/latest summaries.\n",
        f"- Source rows: {len(frame):,}\n"
        f"- Runs/models: {frame.run_id.nunique()}/{frame.model_id.nunique()}\n"
        f"- Measurement types: {frame.measurement_type.nunique()}\n"
        f"- Distinct metric names: {frame.metric_name.nunique()}\n"
        f"- Invalid NaN rows: {int(invalid_nan_rows.sum())} "
        "(all explicitly marked `valid_value=False`)\n",
        "## Run inventory\n",
        _markdown_table(
            inventory,
            [
                "model_family", "model_variant", "model_status", "run_id",
                "run_complete", "target_training_steps", "min_training_step",
                "max_training_step", "metric_count", "measurement_types",
            ],
        ),
        "## Best recorded MAUVE point per model variant\n",
        "These rows are an inventory, not a fair ranking: byte-token and GPT-2 "
        "runs belong to different comparable groups/protocols.\n",
        _markdown_table(best, best_columns),
        "## External generation metrics: core (all canonical points)\n",
        _markdown_table(
            external,
            [
                "model_family", "model_variant", "evaluation_label", "sampler",
                "point", "nfe", "forwards_per_sequence", "n_samples", "seed",
                "mauve", "frontier_integral", "gen_ppl", "token_entropy",
                "scoring_seconds",
            ],
        ),
        "## External generation metrics: distribution diagnostics\n",
        _markdown_table(
            external,
            [
                "model_family", "model_variant", "evaluation_label", "point",
                "seq_rep_2", "seq_rep_3", "seq_rep_4", "distinct_2",
                "distinct_3", "distinct_4", "cross_sample_overlap_4", "zipf",
                "js_1gram", "js_2gram",
            ],
        ),
        "## Sequence latency benchmarks\n",
        _markdown_table(
            latency,
            [
                "model_family", "model_variant", "evaluation_label", "sampler",
                "point", "nfe", "forwards_per_sequence", "sequence_latency_ms",
                "sequence_latency_p10_ms", "sequence_latency_p90_ms",
                "sequence_latency_repeats",
            ],
        ),
        "## Online quality curves\n",
        _markdown_table(
            online,
            [
                "model_family", "model_variant", "training_step", "evaluation_label",
                "point", "nfe", "validation_event", "mauve", "frontier_integral",
                "gen_ppl", "token_entropy", "sampling_seconds", "mauve_seconds",
                "gen_ppl_seconds", "total_seconds",
            ],
        ),
        "## Validation curves\n",
        _markdown_table(
            validation,
            [
                "model_family", "model_variant", "training_step", "epoch",
                "val_loss", "val_nll", "val_nelbo", "val_bits_per_character",
                "val_bits_per_character_estimate", "val_masked_fraction",
                "val_token_count", "val_schedule_candidates",
                "val_selected_mask_rate_min", "val_selected_mask_rate_max",
                "val_selected_schedule_nelbo_variance",
            ],
        ),
        "## Checkpoint metadata\n",
        _markdown_table(
            checkpoint,
            [
                "model_family", "model_variant", "model_status", "run_id",
                "checkpoint_path", "checkpoint_step_x", "checkpoint_step_y",
                "checkpoint_global_step", "checkpoint_epoch", "checkpoint_loadable",
                "checkpoint_best_metric", "checkpoint_ema_decay",
                "checkpoint_parameter_count_total",
                "checkpoint_parameter_count_trainable", "checkpoint_processed_tokens",
                "checkpoint_wall_clock_seconds",
            ],
        ),
        "## Internal checkpoint evaluations\n",
        _markdown_table(
            internal,
            [
                "model_family", "model_variant", "training_step", "test_loss",
                "test_nll", "test_nelbo", "test_perplexity",
                "test_bits_per_character", "test_bits_per_character_estimate",
                "test_epsilon_truncated_nelbo_perplexity_estimate",
                "test_masked_fraction", "test_token_count",
                "likelihood_is_strict_upper_bound", "likelihood_time_min",
                "processed_training_tokens", "sample_adjacent_repetition_rate",
                "sample_distinct_1", "sample_distinct_2",
                "sample_unigram_entropy_nats", "sample_vocabulary_coverage",
                "sampling_seconds", "sampling_tokens_per_second",
            ],
        ),
        "## Generation benchmark\n",
        _markdown_table(
            generation,
            [
                "model_family", "model_variant", "evaluation_label", "point", "nfe",
                "n_samples", "generation_seconds",
            ],
        ),
        "## Every metric by run and measurement type\n",
        "For time-series metrics, `latest_value` is the last valid value ordered by "
        "training/checkpoint step, event time, and source row.\n",
        _markdown_table(
            catalog,
            [
                "model_family", "model_variant", "run_id", "measurement_type",
                "metric_name", "metric_unit", "row_count", "measurement_count",
                "canonical_rows", "duplicate_rows", "min_value", "max_value",
                "mean_value", "latest_value", "min_training_step",
                "max_training_step",
            ],
        ),
    ]
    markdown_path = output_dir / "ALL_METRICS.md"
    markdown_path.write_text("\n".join(report_parts))

    manifest = {
        "source": str(args.input.resolve()),
        "source_rows": len(frame),
        "runs": int(frame.run_id.nunique()),
        "models": int(frame.model_id.nunique()),
        "measurement_types": int(frame.measurement_type.nunique()),
        "metric_names": int(frame.metric_name.nunique()),
        "measurements": int(frame.measurement_id.nunique()),
        "canonical_valid_nonduplicate_rows": len(canonical),
        "invalid_nan_rows": int(invalid_nan_rows.sum()),
        "catalog_rows": len(catalog),
        "tables": table_manifest,
        "artifacts": {
            "run_inventory": str(inventory_path),
            "all_metric_summary": str(catalog_path),
            "canonical_evaluation_wide": str(all_wide_path),
            "best_mauve_points": str(best_path),
            "complete_markdown": str(markdown_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"[done] elapsed={time.perf_counter() - started:.1f}s "
        f"inventory={inventory_path} catalog={catalog_path} "
        f"wide={all_wide_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
