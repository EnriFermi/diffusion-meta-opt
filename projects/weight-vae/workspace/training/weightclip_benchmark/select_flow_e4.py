#!/usr/bin/env python3
"""Select one common solver/NFE pair from four immutable E4 sweep reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_ARMS = {
    ("ours", "gaussian"),
    ("ours", "paired_anchor"),
    ("weightclip", "gaussian"),
    ("weightclip", "paired_anchor"),
}
EXPECTED_GRID = {(solver, steps) for solver in ("euler", "heun") for steps in (4, 8, 16, 32)}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable global E4 selection conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def write_global_e4_selection(*, report_paths: Sequence[Path], output: Path) -> dict[str, Any]:
    paths = [path.resolve() for path in report_paths]
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("global E4 selection requires four unique provisional reports")
    reports = [json.loads(path.read_text()) for path in paths]
    arms = {(str(row.get("codec")), str(row.get("path_kind"))) for row in reports}
    if arms != EXPECTED_ARMS:
        raise ValueError(f"global E4 arms mismatch: observed={sorted(arms)} expected={sorted(EXPECTED_ARMS)}")
    rows_by_arm: dict[tuple[str, str], dict[tuple[str, int], Mapping[str, Any]]] = {}
    for report in reports:
        if int(report.get("schema_version", -1)) != 1 or report.get("kind") != "flow_decoded_e4_sweep":
            raise ValueError("global E4 input is not an immutable provisional sweep")
        arm = (str(report["codec"]), str(report["path_kind"]))
        rows = {(str(row["solver"]), int(row["nfe_steps"])): row for row in report["sweep"]}
        if set(rows) != EXPECTED_GRID:
            raise ValueError(f"incomplete E4 grid for arm {arm}")
        rows_by_arm[arm] = rows
    eligible: list[tuple[float, int, str, int]] = []
    for solver, steps in EXPECTED_GRID:
        rows = [rows_by_arm[arm][(solver, steps)] for arm in sorted(EXPECTED_ARMS)]
        if all(float(row["decoded_finite_fraction"]) == 1.0 for row in rows):
            actual_nfe = steps * (2 if solver == "heun" else 1)
            if any(int(row["actual_nfe"]) != actual_nfe for row in rows):
                raise ValueError("E4 report actual-NFE ledger is inconsistent")
            eligible.append((sum(float(row["endpoint_rmse"]) for row in rows) / 4.0, actual_nfe, solver, steps))
    if not eligible:
        raise RuntimeError("no common E4 candidate passes decoded finite-health gate across all four arms")
    aggregate_rmse, actual_nfe, solver, steps = min(eligible)
    report = {
        "schema_version": 1,
        "kind": "flow_global_e4_selection",
        "selection_policy": "common_four_arm_mean_pooled_endpoint_rmse_then_actual_nfe_then_solver_name",
        "health_gate": "decoded_finite_fraction_equals_one_for_every_arm",
        "reports": [
            {
                "path": str(path),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "codec": report["codec"],
                "path_kind": report["path_kind"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "normalizer_sha256": report["normalizer_sha256"],
            }
            for path, report in sorted(zip(paths, reports), key=lambda item: (item[1]["codec"], item[1]["path_kind"]))
        ],
        "selection": {
            "solver": solver,
            "nfe_steps": steps,
            "actual_nfe": actual_nfe,
            "four_arm_mean_endpoint_rmse": aggregate_rmse,
        },
    }
    _immutable_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_global_e4_selection(report_paths=args.report, output=args.output.resolve())
    print(json.dumps({"stage": "global_e4_selected", "output": str(args.output.resolve()), "selection": report["selection"]}))


if __name__ == "__main__":
    main()
