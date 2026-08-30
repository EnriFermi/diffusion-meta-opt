#!/usr/bin/env python3
"""Re-review existing WeightCLIP benchmark metrics and regenerate tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from big_vae.weightclip_benchmark.reporting import load_metrics, write_evaluation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(f"[review:startup] metrics={args.metrics.resolve()} output={args.output_dir.resolve()}")
    rows = load_metrics(args.metrics)
    provenance = (
        json.loads(args.provenance.read_text(encoding="utf-8"))
        if args.provenance is not None
        else {"review_source": str(args.metrics.resolve())}
    )
    report = write_evaluation_artifacts(
        args.output_dir,
        rows,
        provenance=provenance,
        make_plots=not args.no_plots,
    )
    if not report.valid:
        raise SystemExit("Review failed; inspect review.json")


if __name__ == "__main__":
    main()

