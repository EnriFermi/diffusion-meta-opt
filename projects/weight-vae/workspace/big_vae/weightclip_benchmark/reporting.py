"""Artifact writing and independent validity review for WeightCLIP evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from .contract import CandidateProtocol, DEFAULT_CONTRACT, WeightCLIPBenchmarkContract
from .coverage import validate_exact_grid
from .official_bridge import redact_secrets


@dataclass(frozen=True)
class ReviewIssue:
    severity: str
    code: str
    message: str


@dataclass
class ReviewReport:
    valid: bool
    issues: list[ReviewIssue] = field(default_factory=list)
    row_count: int = 0
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "row_count": self.row_count,
            "issues": [issue.__dict__ for issue in self.issues],
            "files": self.files,
        }


def review_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: WeightCLIPBenchmarkContract = DEFAULT_CONTRACT,
    expected_grid: Sequence[Mapping[str, Any]] | None = None,
) -> ReviewReport:
    issues: list[ReviewIssue] = []
    if expected_grid is not None:
        try:
            validate_exact_grid(rows, expected_grid)
        except ValueError as exc:
            issues.append(ReviewIssue("error", "coverage_grid_mismatch", str(exc)))
    seen: set[tuple[Any, ...]] = set()
    required = {
        "experiment_id", "method", "dataset", "evaluation_seed", "protocol",
        "table_id", "candidate_id", "candidate_budget", "selection_split",
        "head_policy", "head_seed", "batchnorm_policy", "epoch",
        "test_accuracy", "contract_fingerprint",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            issues.append(ReviewIssue("error", "missing_fields", f"row {index}: missing {missing}"))
            continue
        try:
            contract.assert_matches(row)
        except ValueError as exc:
            issues.append(ReviewIssue("error", "contract_mismatch", f"row {index}: {exc}"))
        protocol = str(row["protocol"])
        if str(row["table_id"]) != protocol:
            issues.append(ReviewIssue("error", "mixed_table", f"row {index}: table_id must equal protocol"))
        _check_protocol_fields(row, index, issues, contract)
        key = (
            row["experiment_id"], row["method"], row["dataset"], row["evaluation_seed"],
            protocol, row["candidate_id"], row["epoch"],
        )
        if key in seen:
            issues.append(ReviewIssue("error", "duplicate_row", f"duplicate metric identity: {key}"))
        seen.add(key)
        for metric in ("test_accuracy", "validation_accuracy", "train_loss", "aulc"):
            value = row.get(metric)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                issues.append(ReviewIssue("error", "nonnumeric_metric", f"row {index}: {metric}={value!r}"))
                continue
            if not math.isfinite(number):
                issues.append(ReviewIssue("error", "nonfinite_metric", f"row {index}: {metric}={number}"))
            if metric.endswith("accuracy") and not 0.0 <= number <= 1.0:
                issues.append(ReviewIssue("error", "impossible_accuracy", f"row {index}: {metric}={number}"))

    paired_head_seeds: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        if row.get("head_policy") == contract.access.classifier_head_policy:
            group = (row.get("dataset"), row.get("evaluation_seed"), row.get("protocol"), row.get("candidate_id"), row.get("epoch"))
            paired_head_seeds[group].add(int(row["head_seed"]))
    for group, seeds in paired_head_seeds.items():
        if len(seeds) > 1:
            issues.append(ReviewIssue("error", "mismatched_head_seed", f"group {group} has head seeds {sorted(seeds)}"))

    run_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if required <= set(row):
            group = (
                row["experiment_id"], row["method"], row["dataset"],
                row["evaluation_seed"], row["protocol"],
            )
            run_groups[group].append(row)
    for group, group_rows in run_groups.items():
        epoch_zero = [row for row in group_rows if int(row["epoch"]) == 0]
        budgets = {int(row["candidate_budget"]) for row in group_rows}
        if len(budgets) != 1:
            issues.append(ReviewIssue("error", "inconsistent_candidate_budget", f"group {group} budgets={sorted(budgets)}"))
            continue
        budget = next(iter(budgets))
        if len(epoch_zero) != budget:
            issues.append(ReviewIssue("error", "missing_candidate_rows", f"group {group} has {len(epoch_zero)} epoch-0 candidates, expected {budget}"))
        selected_ids = {str(row["candidate_id"]) for row in epoch_zero if bool(row.get("selected", False))}
        expected_selected = 5 if str(group[-1]) == CandidateProtocol.NATIVE_TEST_TOP5_ORACLE.value else 1
        if len(selected_ids) != expected_selected:
            issues.append(ReviewIssue("error", "selected_count_mismatch", f"group {group} selected {len(selected_ids)}, expected {expected_selected}"))
        for candidate_id in selected_ids:
            epochs = {int(row["epoch"]) for row in group_rows if str(row["candidate_id"]) == candidate_id}
            missing_epochs = set(contract.evaluation.report_epochs) - epochs
            if missing_epochs:
                issues.append(ReviewIssue("error", "missing_report_epochs", f"group {group} candidate={candidate_id} missing epochs {sorted(missing_epochs)}"))

    return ReviewReport(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        row_count=len(rows),
    )


def write_evaluation_artifacts(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    contract: WeightCLIPBenchmarkContract = DEFAULT_CONTRACT,
    make_plots: bool = True,
    expected_grid: Sequence[Mapping[str, Any]] | None = None,
) -> ReviewReport:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[reporting:start] stage=review rows={len(rows)} output={output}")
    review = review_rows(rows, contract, expected_grid)
    clean_rows = [redact_secrets(dict(row)) for row in rows]
    clean_provenance = redact_secrets(dict(provenance))
    paths: dict[str, Path] = {}
    paths["metrics_jsonl"] = output / "metrics.jsonl"
    paths["metrics_jsonl"].write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in clean_rows), encoding="utf-8")
    paths["metrics_csv"] = output / "metrics.csv"
    _write_csv(paths["metrics_csv"], clean_rows)
    summaries = aggregate_rows(clean_rows)
    paths["summary_json"] = output / "summary.json"
    paths["summary_json"].write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["provenance"] = output / "provenance.json"
    paths["provenance"].write_text(
        json.dumps(
            clean_provenance
            | {
                "contract_version": contract.version,
                "contract_fingerprint": contract.fingerprint(),
                "contract": contract.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for protocol in CandidateProtocol:
        table = [item for item in summaries if item["protocol"] == protocol.value]
        if not table:
            continue
        csv_path = output / f"table_{protocol.value}.csv"
        md_path = output / f"table_{protocol.value}.md"
        _write_csv(csv_path, table)
        _write_markdown(md_path, table)
        paths[f"table_{protocol.value}_csv"] = csv_path
        paths[f"table_{protocol.value}_md"] = md_path
    if make_plots and clean_rows:
        plot_path = output / "fine_tuning_curves.png"
        try:
            _plot_curves(plot_path, clean_rows)
            _inspect_plot(plot_path, review.issues)
            paths["curves"] = plot_path
        except ImportError:
            review.issues.append(ReviewIssue("warning", "plot_dependency_missing", "matplotlib is unavailable"))
    review.valid = not any(issue.severity == "error" for issue in review.issues)
    review.files = {name: str(path) for name, path in paths.items()}
    review_payload = review.to_dict()
    review_payload["sha256"] = {name: _sha256(path) for name, path in paths.items()}
    review_path = output / "review.json"
    review_path.write_text(json.dumps(review_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review.files["review"] = str(review_path)
    print(
        f"[reporting:done] valid={review.valid} errors={sum(i.severity == 'error' for i in review.issues)} "
        f"warnings={sum(i.severity == 'warning' for i in review.issues)} review={review_path}"
    )
    for name, path in review.files.items():
        print(f"[reporting:artifact] {name}={path}")
    return review


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        # Scientific/native tables summarize the actually selected estimator.
        # All-candidate epoch-0 distributions remain available in metrics.csv
        # and the per-run summary, but must not replace native top-5 reporting.
        if not bool(row.get("selected", False)):
            continue
        key = (str(row["protocol"]), str(row["method"]), str(row["dataset"]), int(row["epoch"]))
        groups[key].append(float(row["test_accuracy"]))
    output = []
    for (protocol, method, dataset, epoch), values in sorted(groups.items()):
        output.append(
            {
                "protocol": protocol,
                "method": method,
                "dataset": dataset,
                "epoch": epoch,
                "n": len(values),
                "test_accuracy_mean": fmean(values),
                "test_accuracy_std": pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    return output


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return list(payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload)
    if source.suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported metrics format: {source}")


def _check_protocol_fields(
    row: Mapping[str, Any],
    index: int,
    issues: list[ReviewIssue],
    contract: WeightCLIPBenchmarkContract,
) -> None:
    protocol = str(row["protocol"])
    budget = int(row["candidate_budget"])
    split = str(row["selection_split"])
    expected: dict[str, tuple[int, str]] = {
        CandidateProtocol.CONTROLLED_SINGLE.value: (1, "none_random_precommitted"),
        CandidateProtocol.CONTROLLED_VALIDATION_BEST_K.value: (contract.evaluation.controlled_candidate_count, "validation"),
        CandidateProtocol.NATIVE_TEST_TOP5_ORACLE.value: (contract.evaluation.native_candidate_count, "test"),
    }
    if protocol not in expected:
        issues.append(ReviewIssue("error", "unknown_protocol", f"row {index}: {protocol!r}"))
        return
    expected_budget, expected_split = expected[protocol]
    if budget != expected_budget or split != expected_split:
        issues.append(
            ReviewIssue(
                "error",
                "candidate_budget_mismatch",
                f"row {index}: {protocol} requires budget/split {expected_budget}/{expected_split}, got {budget}/{split}",
            )
        )
    if protocol == CandidateProtocol.NATIVE_TEST_TOP5_ORACLE.value and int(row.get("selected_count_budget", -1)) != 5:
        issues.append(ReviewIssue("error", "native_topk_mismatch", f"row {index}: native oracle must select top 5"))
    if (
        protocol == CandidateProtocol.CONTROLLED_VALIDATION_BEST_K.value
        and int(row.get("selected_count_budget", -1)) != contract.evaluation.controlled_top_k
    ):
        issues.append(ReviewIssue("error", "controlled_topk_mismatch", f"row {index}: controlled secondary must select top 5"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["method", "dataset", "epoch", "n", "test_accuracy_mean", "test_accuracy_std"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _plot_curves(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["epoch"]) > 0 and not bool(row.get("selected", False)):
            continue
        groups[(str(row["protocol"]), str(row["method"]))][int(row["epoch"])].append(float(row["test_accuracy"]))
    protocols = sorted({key[0] for key in groups})
    fig, axes = plt.subplots(1, len(protocols), figsize=(max(7, 6 * len(protocols)), 5), squeeze=False)
    for axis, protocol in zip(axes[0], protocols):
        for (group_protocol, method), points in sorted(groups.items()):
            if group_protocol != protocol:
                continue
            epochs = sorted(points)
            means = [fmean(points[epoch]) for epoch in epochs]
            axis.plot(epochs, means, marker="o", label=method)
        axis.set_title(protocol.replace("_", " "))
        axis.set_xlabel("Fine-tuning epoch")
        axis.set_ylabel("Test accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("WeightCLIP benchmark: protocol-separated fine-tuning curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _inspect_plot(path: Path, issues: list[ReviewIssue]) -> None:
    if not path.exists() or path.stat().st_size < 10_000:
        issues.append(ReviewIssue("error", "plot_unreadable", f"plot absent or suspiciously small: {path}"))
        return
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        if width < 900 or height < 600:
            issues.append(ReviewIssue("warning", "plot_low_resolution", f"plot is only {width}x{height}: {path}"))
    except ImportError:
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
