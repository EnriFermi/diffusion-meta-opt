from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    allowed_pairs,
    coverage_report,
    expected_datasets,
    expected_models,
    pair_label,
    write_json,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)!r}")
    return payload


def _resolve_path(path: str | Path) -> Path:
    value = Path(str(path)).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value


def _latest_eval_dir(heldout_root: Path) -> Path:
    eval_root = heldout_root / "eval"
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")
    candidates = [path for path in eval_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No eval run directories under {eval_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _split_pair_label(value: str) -> tuple[str, str] | None:
    text = str(value).strip()
    if "::" not in text:
        return None
    dataset_name, model_name = text.split("::", 1)
    dataset_name = dataset_name.strip()
    model_name = model_name.strip()
    if not dataset_name or not model_name:
        return None
    return dataset_name, model_name


def _coverage_from_existing_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    coverage = payload.get("coverage")
    if isinstance(coverage, dict):
        return dict(coverage)
    return None


def _coverage_from_pair_rows(rows: Iterable[dict[str, str]], *, skipped: dict[str, int] | None = None) -> dict[str, Any]:
    observed_pairs: set[tuple[str, str]] = set()
    observed_datasets: set[str] = set()
    observed_models: set[str] = set()
    for row in rows:
        dataset_name = str(row.get("dataset", "")).strip()
        model_name = str(row.get("model", "")).strip()
        if not dataset_name or not model_name:
            continue
        observed_datasets.add(dataset_name)
        observed_models.add(model_name)
        observed_pairs.add((dataset_name, model_name))
    return coverage_report(
        observed_datasets=observed_datasets,
        observed_models=observed_models,
        observed_pairs=observed_pairs,
        skipped=skipped or {},
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def check_eval_dir(eval_dir: Path) -> dict[str, Any]:
    eval_dir = _resolve_path(eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    coverage_path = eval_dir / "coverage.json"
    if coverage_path.exists():
        coverage = _read_json(coverage_path)
        source = str(coverage_path)
    else:
        summary_path = eval_dir / "metrics_summary.json"
        coverage = None
        source = ""
        if summary_path.exists():
            summary_payload = _read_json(summary_path)
            coverage = _coverage_from_existing_payload(summary_payload)
            source = str(summary_path)

        if coverage is None:
            pair_csv = eval_dir / "metrics_by_dataset_model_pair.csv"
            record_csv = eval_dir / "record_metrics.csv"
            if pair_csv.exists():
                rows = _read_csv_rows(pair_csv)
                coverage = _coverage_from_pair_rows(rows)
                source = str(pair_csv)
            elif record_csv.exists():
                rows = _read_csv_rows(record_csv)
                non_finite = sum(1 for row in rows if str(row.get("finite", "")).strip().lower() not in {"true", "1", "yes"})
                coverage = _coverage_from_pair_rows(rows, skipped={"non_finite": non_finite})
                source = str(record_csv)
                coverage["partial_source_warning"] = (
                    "coverage inferred from record_metrics.csv; if eval is still running, rows can be buffered"
                )
            else:
                raise FileNotFoundError(
                    f"Could not infer coverage under {eval_dir}; expected coverage.json, "
                    "metrics_summary.json, metrics_by_dataset_model_pair.csv, or record_metrics.csv"
                )

    return {
        "kind": "eval",
        "path": str(eval_dir),
        "source": source,
        "coverage": coverage,
    }


def check_build_root(heldout_root: Path) -> dict[str, Any]:
    heldout_root = _resolve_path(heldout_root)
    summary_path = heldout_root / "heldout_build_analysis" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Build summary not found: {summary_path}")
    payload = _read_json(summary_path)
    coverage = _coverage_from_existing_payload(payload)
    if coverage is None:
        pair_counts_raw = payload.get("pair_counts", {})
        dataset_counts_raw = payload.get("dataset_counts", {})
        model_counts_raw = payload.get("model_counts", {})
        observed_pairs = {
            pair
            for pair in (_split_pair_label(key) for key, value in dict(pair_counts_raw).items() if int(value) > 0)
            if pair is not None
        }
        coverage = coverage_report(
            observed_datasets={str(key) for key, value in dict(dataset_counts_raw).items() if int(value) > 0},
            observed_models={str(key) for key, value in dict(model_counts_raw).items() if int(value) > 0},
            observed_pairs=observed_pairs,
            skipped={},
        )
    return {
        "kind": "build",
        "path": str(heldout_root),
        "source": str(summary_path),
        "coverage": coverage,
    }


def _print_human(report: dict[str, Any]) -> None:
    coverage = dict(report["coverage"])
    print(f"{report['kind']} coverage source: {report['source']}")
    print(f"coverage.ok = {coverage.get('ok')}")
    print(
        "datasets: "
        f"{coverage.get('observed_dataset_count', 0)}/{coverage.get('expected_dataset_count', len(expected_datasets()))}"
    )
    print(
        "models: "
        f"{coverage.get('observed_model_count', 0)}/{coverage.get('expected_model_count', len(expected_models()))}"
    )
    print(
        "pairs: "
        f"{coverage.get('observed_pair_count', 0)}/{coverage.get('expected_pair_count', len(allowed_pairs()))}"
    )
    print(f"skipped_total = {coverage.get('skipped_total', 0)}")
    for key in (
        "missing_datasets",
        "missing_models",
        "missing_pairs",
        "unexpected_datasets",
        "unexpected_models",
        "unexpected_pairs",
    ):
        values = coverage.get(key, [])
        if values:
            print(f"{key}: {values}")
    skipped = coverage.get("skipped", {})
    if skipped:
        print(f"skipped: {skipped}")
    if coverage.get("partial_source_warning"):
        print(f"warning: {coverage['partial_source_warning']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-facto held-out BigVAE coverage checker.")
    parser.add_argument("--heldout-root", default=os.environ.get("HELDOUT_ROOT", ""), help="Held-out offline dataset root.")
    parser.add_argument("--eval-dir", default=os.environ.get("EVAL_OUTPUT_DIR", ""), help="Eval output directory.")
    parser.add_argument("--latest-eval", action="store_true", help="Use latest directory under HELDOUT_ROOT/eval.")
    parser.add_argument("--build", action="store_true", help="Check build coverage instead of eval coverage.")
    parser.add_argument("--json-out", default="", help="Optional path to write coverage report JSON.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0.")
    args = parser.parse_args()

    if args.build:
        if not args.heldout_root:
            raise SystemExit("--heldout-root or HELDOUT_ROOT is required with --build")
        report = check_build_root(_resolve_path(args.heldout_root))
    else:
        if args.eval_dir:
            eval_dir = _resolve_path(args.eval_dir)
        else:
            if not args.heldout_root:
                raise SystemExit("--eval-dir, EVAL_OUTPUT_DIR, --heldout-root, or HELDOUT_ROOT is required")
            eval_dir = _latest_eval_dir(_resolve_path(args.heldout_root))
        report = check_eval_dir(eval_dir)

    _print_human(report)
    if args.json_out:
        write_json(_resolve_path(args.json_out), report)
    if not args.no_fail and not bool(report["coverage"].get("ok", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
