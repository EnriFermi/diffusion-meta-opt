from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import default_heldout_eval_dir, default_heldout_root

PROGRESS_RE = re.compile(
    r"Held-out build progress: seen_total=(?P<seen>\d+) accepted=(?P<accepted>\d+) "
    r"size_gb=(?P<size>[0-9.]+) pair_min=(?P<pair_min>\d+) pair_max=(?P<pair_max>\d+)"
)
MISSING_RE = re.compile(r"missing_pairs=(?P<missing>\[[^\n]*?\])")
WAIT_RE = re.compile(r"Waiting for held-out sample: waited_s=(?P<waited>[0-9.]+) context=(?P<context>\{.*?\}) runtime=")
MODEL_FAILURE_RE = re.compile(
    r"Disabling collector model after runtime failure: model=(?P<model>\S+) error=(?P<error>.*)"
)
ERROR_MARKERS = (
    "Disabling collector model",
    "Collector model remains disabled",
    "Collector job failed",
    "collector job failed",
    "failed model",
    "model failed",
    "Traceback",
    "RuntimeError",
    "Exception",
    "ERROR",
    "CUDA",
    "out of memory",
)
HELDOUT_EXPECTED_MODELS = {
    "clip_vit_l14",
    "detr_resnet50",
    "donut_rvlcdip",
    "segformer_b5_cityscapes",
    "siglip_so400m_p14_384",
    "trocr_large_printed",
    "vit_base_p16_224",
    "vit_large_p16_224",
}


def _line_contains_error_marker(line: str) -> bool:
    lowered = line.lower()
    return any(marker in line or marker.lower() in lowered for marker in ERROR_MARKERS)


def _models_in_text(text: str) -> list[str]:
    return sorted(model for model in HELDOUT_EXPECTED_MODELS if model in text)


def _make_log_snippet(
    lines: list[str],
    *,
    idx: int,
    snippet_lines: int,
) -> dict[str, Any]:
    start = max(0, int(idx) - 3)
    stop = min(len(lines), int(idx) + max(1, int(snippet_lines)))
    snippet = lines[start:stop]
    return {
        "line_number": int(start + 1),
        "header": lines[int(idx)] if 0 <= int(idx) < len(lines) else "",
        "models": _models_in_text("\n".join(snippet)),
        "snippet": snippet,
    }
HELDOUT_EXPECTED_DATASETS = {
    "chexpert",
    "flickr30k",
    "food101",
    "openimages_v7",
    "pascal_voc_2012",
    "rvl_cdip",
    "sun397",
}


def _env_path(name: str, default: str | Path) -> Path:
    raw = os.environ.get(name)
    value = str(default) if raw is None or str(raw).strip() == "" else str(raw).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = _resolve_path(path)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _first_existing(paths: list[Path]) -> Path | None:
    for path in _dedupe_paths(paths):
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)!r}")
    return payload


def _safe_jsonl_last(path: Path, limit: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    for line in lines[-max(1, int(limit)) :]:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        except Exception:
            continue
    return rows


def _resolve_latest_log(log_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for log_dir in _dedupe_paths(log_dirs):
        if log_dir.exists():
            candidates.extend(log_dir.glob("build_big_vae_heldout_offline_dataset*.log"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _parse_log(log_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "log_path": str(log_path),
        "last_progress": None,
        "last_missing_pairs": [],
        "last_wait": None,
        "model_failures": [],
        "tail_errors": [],
        "tail_warnings": [],
    }
    if not log_path.exists():
        return payload

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        progress = PROGRESS_RE.search(line)
        if progress:
            payload["last_progress"] = {
                "seen_total": int(progress.group("seen")),
                "accepted": int(progress.group("accepted")),
                "size_gb": float(progress.group("size")),
                "pair_min": int(progress.group("pair_min")),
                "pair_max": int(progress.group("pair_max")),
            }
            missing = MISSING_RE.search(line)
            if missing:
                try:
                    payload["last_missing_pairs"] = ast.literal_eval(missing.group("missing"))
                except Exception:
                    payload["last_missing_pairs"] = [missing.group("missing")]

        wait = WAIT_RE.search(line)
        if wait:
            payload["last_wait"] = {"waited_s": float(wait.group("waited")), "raw": line}

        model_failure = MODEL_FAILURE_RE.search(line)
        if model_failure:
            payload["model_failures"].append(
                {
                    "model": model_failure.group("model"),
                    "error": model_failure.group("error"),
                    "raw": line,
                }
            )
            payload["model_failures"] = payload["model_failures"][-20:]

        if " ERROR " in line or " failed " in line.lower() or "fatal" in line.lower():
            payload["tail_errors"].append(line)
            payload["tail_errors"] = payload["tail_errors"][-20:]
        elif " WARNING " in line:
            payload["tail_warnings"].append(line)
            payload["tail_warnings"] = payload["tail_warnings"][-20:]
    return payload


def _is_heldout_load_summary(summary: dict[str, Any]) -> bool:
    path = str(summary.get("path", ""))
    if "heldout" in path:
        return True
    expected = summary.get("expected", {})
    if not isinstance(expected, dict):
        return False
    models = set(str(item) for item in expected.get("models", []) or [])
    datasets = set(str(item) for item in expected.get("datasets", []) or [])
    return HELDOUT_EXPECTED_MODELS.issubset(models) and HELDOUT_EXPECTED_DATASETS.issubset(datasets)


def _collect_load_reports(report_dirs: list[Path], limit_events: int = 80) -> dict[str, Any]:
    summary_paths: list[Path] = []
    event_paths: list[Path] = []
    for report_dir in _dedupe_paths(report_dirs):
        if not report_dir.exists():
            continue
        summary_paths.extend(report_dir.rglob("*load_summary.json"))
        event_paths.extend(report_dir.rglob("*load_report.jsonl"))

    summaries: list[dict[str, Any]] = []
    for path in sorted(set(summary_paths), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = _read_json(path)
        except Exception as exc:
            summaries.append({"path": str(path), "error": f"read_failed: {exc}"})
            continue
        summaries.append(
            {
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "expected": payload.get("expected", {}),
                "loaded": payload.get("loaded", {}),
                "failed": payload.get("failed", {}),
                "pending": payload.get("pending", {}),
            }
        )
    heldout_summaries = [summary for summary in summaries if _is_heldout_load_summary(summary)]

    events: list[dict[str, Any]] = []
    interesting_events = {
        "dataset_failed",
        "dataset_loaded",
        "model_failed",
        "model_loaded",
        "runtime_init_failed",
        "runtime_init_ready",
        "runtime_init_start",
        "expected",
    }
    for path in sorted(set(event_paths), key=lambda p: p.stat().st_mtime, reverse=True):
        if "heldout" not in str(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            events.append({"path": str(path), "error": f"read_failed: {exc}"})
            continue
        for line in lines[-max(1, int(limit_events)) :]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("event", "")) not in interesting_events:
                continue
            row = dict(row)
            row["path"] = str(path)
            events.append(row)
    return {
        "report_dirs": [str(path) for path in _dedupe_paths(report_dirs)],
        "summary_count": len(summary_paths),
        "event_file_count": len(event_paths),
        "summaries": heldout_summaries[:12],
        "ignored_non_heldout_summary_count": max(0, len(summaries) - len(heldout_summaries)),
        "events_tail": events[-max(1, int(limit_events)) :],
    }


def _collect_process_logs(
    log_dirs: list[Path],
    limit_files: int = 16,
    tail_lines: int = 160,
    snippet_lines: int = 40,
    max_snippets: int = 32,
) -> dict[str, Any]:
    patterns = [
        "collector_process*.log",
        "*collector_process*.stdout.log",
        "*collector_process*.stderr.log",
        "*collector_process*_fault*.log",
        "dataset_worker_rvl_cdip*.log",
        "*dataset_worker_rvl_cdip*.stdout.log",
        "*dataset_worker_rvl_cdip*.stderr.log",
    ]
    candidates: list[Path] = []
    for log_dir in _dedupe_paths(log_dirs):
        if not log_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(log_dir.rglob(pattern))
    unique = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    files: list[dict[str, Any]] = []
    for path in unique[: max(1, int(limit_files))]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            files.append({"path": str(path), "error": f"read_failed: {exc}"})
            continue
        interesting = [
            line
            for line in lines
            if (
                "Disabling collector model" in line
                or "Collector model remains disabled" in line
                or "Traceback" in line
                or "RuntimeError" in line
                or "Exception" in line
                or "ERROR" in line
                or "WARNING" in line
                or "CUDA" in line
                or "out of memory" in line.lower()
                or "trust_remote_code" in line
                or "rvl_cdip" in line
            )
        ]
        model_mention_counts = Counter(
            model for line in lines for model in HELDOUT_EXPECTED_MODELS if model in line
        )
        error_snippets: list[dict[str, Any]] = []
        model_failure_snippets: list[dict[str, Any]] = []
        seen_snippets: set[tuple[int, str]] = set()
        for idx, line in enumerate(lines):
            if not _line_contains_error_marker(line):
                continue
            snippet_payload = _make_log_snippet(lines, idx=idx, snippet_lines=snippet_lines)
            error_snippets.append(snippet_payload)
            if len(error_snippets) > max(1, int(max_snippets)):
                error_snippets = error_snippets[-max(1, int(max_snippets)) :]
            models = list(snippet_payload.get("models", []))
            if not models and "Disabling collector model after runtime failure" not in line:
                continue
            key = (int(snippet_payload.get("line_number", 0)), line)
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            model_failure_snippets.append(snippet_payload)
            if len(model_failure_snippets) >= max(1, int(max_snippets)):
                break
        exact_failure_lines: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            match = MODEL_FAILURE_RE.search(line)
            if match is None:
                continue
            exact_failure_lines.append(
                {
                    "line_number": int(idx + 1),
                    "model": str(match.group("model")).strip(),
                    "error": str(match.group("error")).strip(),
                    "line": line,
                }
            )
        files.append(
            {
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "line_count": len(lines),
                "model_mention_counts": {str(key): int(value) for key, value in sorted(model_mention_counts.items())},
                "exact_model_failure_lines": exact_failure_lines,
                "error_snippets": error_snippets,
                "model_failure_snippets": model_failure_snippets,
                "interesting_tail": interesting[-max(1, int(tail_lines)) :],
                "tail": lines[-min(len(lines), max(1, int(tail_lines))) :],
            }
        )
    return {
        "log_dirs": [str(path) for path in _dedupe_paths(log_dirs)],
        "matched_file_count": len(unique),
        "files": files,
    }


def _summarize_worker_status(worker_status_dir: Path, limit: int = 12) -> dict[str, Any]:
    all_path = worker_status_dir / "_all_workers.jsonl"
    if not all_path.exists():
        return {"worker_status_dir": str(worker_status_dir), "exists": False}

    last_by_dataset: dict[str, dict[str, Any]] = {}
    error_counts: Counter[str] = Counter()
    rows_seen = 0
    with all_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rows_seen += 1
            dataset = str(row.get("dataset", "<unknown_dataset>"))
            last_by_dataset[dataset] = row
            error = str(row.get("worker_last_error") or "").strip()
            if error:
                error_counts[f"{dataset}: {error}"] += 1

    problem_rows = []
    for dataset, row in sorted(last_by_dataset.items()):
        if row.get("worker_last_error") or row.get("worker_permanently_stopped") or row.get("worker_exitcode") not in (None, 0):
            problem_rows.append(
                {
                    "dataset": dataset,
                    "worker_alive": row.get("worker_alive"),
                    "worker_exitcode": row.get("worker_exitcode"),
                    "worker_permanently_stopped": row.get("worker_permanently_stopped"),
                    "worker_restarts": row.get("worker_restarts"),
                    "samples_served": row.get("samples_served"),
                    "chunks_ready_total": row.get("chunks_ready_total"),
                    "last_error": row.get("worker_last_error"),
                }
            )

    return {
        "worker_status_dir": str(worker_status_dir),
        "exists": True,
        "rows_seen": rows_seen,
        "problem_workers": problem_rows[: max(1, int(limit))],
        "top_worker_errors": [{"error": key, "count": count} for key, count in error_counts.most_common(limit)],
        "last_by_dataset": {
            dataset: {
                "worker_alive": row.get("worker_alive"),
                "worker_exitcode": row.get("worker_exitcode"),
                "worker_permanently_stopped": row.get("worker_permanently_stopped"),
                "worker_restarts": row.get("worker_restarts"),
                "samples_served": row.get("samples_served"),
                "chunks_ready_total": row.get("chunks_ready_total"),
                "worker_last_error": row.get("worker_last_error"),
            }
            for dataset, row in sorted(last_by_dataset.items())
        },
    }


def _collect_fatal_reports(crashes_dir: Path, limit: int = 12) -> list[dict[str, Any]]:
    candidates = sorted(crashes_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports: list[dict[str, Any]] = []
    for path in candidates[: max(1, int(limit))]:
        try:
            payload = _read_json(path)
        except Exception:
            continue
        reports.append(
            {
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "role": payload.get("role"),
                "error": payload.get("error"),
                "traceback_last_line": str(payload.get("traceback", "")).strip().splitlines()[-1]
                if payload.get("traceback")
                else "",
                "extra": payload.get("extra"),
            }
        )
    return reports


def _summarize_crash_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = _read_json(path)
    async_last_status = payload.get("async_last_status")
    if not isinstance(async_last_status, dict):
        async_last_status = {}
    return {
        "path": str(path),
        "exists": True,
        "exit_code": payload.get("exit_code"),
        "exit_reason": payload.get("exit_reason"),
        "collector_pid": payload.get("collector_pid"),
        "jobs_total": payload.get("jobs_total"),
        "items_emitted": payload.get("items_emitted"),
        "jobs_by_model": payload.get("jobs_by_model"),
        "failed_models": payload.get("failed_models", async_last_status.get("failed_models", [])),
        "failed_model_errors": payload.get("failed_model_errors", async_last_status.get("failed_model_errors", {})),
        "async_recent_jobs_tail": list(payload.get("async_recent_jobs", []))[-5:]
        if isinstance(payload.get("async_recent_jobs"), list)
        else [],
        "fatal_error": async_last_status.get("fatal_error"),
        "fatal_traceback_last_line": str(async_last_status.get("fatal_traceback", "")).strip().splitlines()[-1]
        if async_last_status.get("fatal_traceback")
        else "",
        "fatal_report_path": async_last_status.get("fatal_report_path"),
        "prepared_job_producer_error": async_last_status.get("prepared_job_producer_error"),
        "dataset_workers_top_by_rss": payload.get("dataset_workers_top_by_rss", []),
        "cgroup": payload.get("cgroup", {}),
    }


def _extract_model_failures_from_process_logs(process_logs: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not isinstance(process_logs, dict):
        return out
    for file_payload in process_logs.get("files", []) or []:
        if not isinstance(file_payload, dict):
            continue
        path = str(file_payload.get("path", ""))
        for payload in file_payload.get("exact_model_failure_lines", []) or []:
            if not isinstance(payload, dict):
                continue
            model = str(payload.get("model", "")).strip()
            if not model:
                continue
            out.setdefault(model, []).append(f"{path}:{payload.get('line_number')}: {payload.get('line')}")
        for snippet_payload in file_payload.get("model_failure_snippets", []) or []:
            if not isinstance(snippet_payload, dict):
                continue
            header = str(snippet_payload.get("header", ""))
            lines = snippet_payload.get("snippet", [])
            snippet_text = "\n".join(str(line) for line in lines) if isinstance(lines, list) else ""
            match = MODEL_FAILURE_RE.search(header)
            if match is None:
                continue
            model = str(match.group("model")).strip()
            if not model:
                continue
            out.setdefault(model, []).append(f"{path}:{snippet_payload.get('line_number')}: " + snippet_text)
        for line in file_payload.get("interesting_tail", []) or []:
            if not isinstance(line, str):
                continue
            match = MODEL_FAILURE_RE.search(line)
            if match is None:
                continue
            model = str(match.group("model")).strip()
            if model:
                out.setdefault(model, []).append(f"{path}: {line}")
    return out


def _extract_general_error_snippets(process_logs: dict[str, Any], limit: int = 16) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(process_logs, dict):
        return out
    for file_payload in process_logs.get("files", []) or []:
        if not isinstance(file_payload, dict):
            continue
        path = str(file_payload.get("path", ""))
        for snippet_payload in file_payload.get("error_snippets", []) or []:
            if not isinstance(snippet_payload, dict):
                continue
            out.append(
                {
                    "path": path,
                    "line_number": snippet_payload.get("line_number"),
                    "header": snippet_payload.get("header"),
                    "models": snippet_payload.get("models", []),
                    "snippet": snippet_payload.get("snippet", []),
                }
            )
    return out[-max(1, int(limit)) :]


def _extract_dataset_mentions_from_process_logs(process_logs: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not isinstance(process_logs, dict):
        return out
    for file_payload in process_logs.get("files", []) or []:
        if not isinstance(file_payload, dict):
            continue
        path = str(file_payload.get("path", ""))
        for line in file_payload.get("interesting_tail", []) or []:
            if not isinstance(line, str):
                continue
            for dataset in HELDOUT_EXPECTED_DATASETS:
                if dataset in line:
                    out.setdefault(dataset, []).append(f"{path}: {line}")
    return out


def _build_failure_attribution(report: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    datasets: dict[str, dict[str, Any]] = {}

    log = report.get("log", {})
    if isinstance(log, dict):
        for item in log.get("model_failures", []) or []:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            if model:
                models.setdefault(model, {})["log_failure"] = item.get("error") or item.get("raw")

    crash = report.get("collector_crash_report", {})
    if isinstance(crash, dict):
        failed_model_errors = crash.get("failed_model_errors")
        if isinstance(failed_model_errors, dict):
            for model, payload in failed_model_errors.items():
                row = models.setdefault(str(model), {})
                row["crash_error"] = payload
        failed_models = crash.get("failed_models")
        if isinstance(failed_models, list):
            for model in failed_models:
                models.setdefault(str(model), {})["crash_failed_model_without_traceback"] = True
        jobs_by_model = crash.get("jobs_by_model")
        if isinstance(jobs_by_model, dict):
            for model, jobs in jobs_by_model.items():
                models.setdefault(str(model), {})["jobs"] = jobs

    load_reports = report.get("load_reports", {})
    if isinstance(load_reports, dict):
        for summary in load_reports.get("summaries", []) or []:
            if not isinstance(summary, dict):
                continue
            failed = summary.get("failed", {})
            if not isinstance(failed, dict):
                continue
            failed_models = failed.get("models", {})
            if isinstance(failed_models, dict):
                for model, payload in failed_models.items():
                    models.setdefault(str(model), {})["load_report_failure"] = payload
            failed_datasets = failed.get("datasets", {})
            if isinstance(failed_datasets, dict):
                for dataset, payload in failed_datasets.items():
                    datasets.setdefault(str(dataset), {})["load_report_failure"] = payload

    process_logs = report.get("process_logs", {})
    for model, lines in _extract_model_failures_from_process_logs(process_logs).items():
        models.setdefault(str(model), {})["process_log_failures"] = lines[-5:]
    for dataset, lines in _extract_dataset_mentions_from_process_logs(process_logs).items():
        datasets.setdefault(str(dataset), {})["process_log_mentions"] = lines[-5:]

    worker_status = report.get("worker_status", {})
    if isinstance(worker_status, dict):
        last_by_dataset = worker_status.get("last_by_dataset", {})
        if isinstance(last_by_dataset, dict):
            for dataset, payload in last_by_dataset.items():
                if not isinstance(payload, dict):
                    continue
                if (
                    payload.get("worker_last_error")
                    or payload.get("worker_permanently_stopped")
                    or payload.get("worker_alive") is False
                ):
                    datasets.setdefault(str(dataset), {})["worker_status"] = payload

    return {
        "models": models,
        "datasets": datasets,
        "process_log_errors": _extract_general_error_snippets(process_logs),
        "note": (
            "If a failed model only appears by name without an error payload, the old run did not persist "
            "the traceback. process_log_errors is the bounded proof set from saved collector/dataset logs; "
            "if it has no model failure traceback, the traceback is absent from the copied artifacts."
        ),
    }


def _print_section(title: str, value: Any) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose held-out BigVAE offline dataset build failures.")
    parser.add_argument("--heldout-root", default=os.environ.get("HELDOUT_ROOT", ""))
    parser.add_argument("--log-dir", default=os.environ.get("HELDOUT_LOG_DIR", ""))
    parser.add_argument("--log-file", default="")
    parser.add_argument("--reports-dir", default="")
    parser.add_argument("--crashes-dir", default="")
    parser.add_argument("--crash-report", default="")
    parser.add_argument("--worker-status-dir", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    heldout_root = _env_path(
        "HELDOUT_ROOT",
        args.heldout_root or default_heldout_root(),
    )
    heldout_eval_dir = default_heldout_eval_dir()
    log_dirs = _dedupe_paths(
        [
            _resolve_path(args.log_dir) if args.log_dir else _env_path("HELDOUT_LOG_DIR", heldout_eval_dir / "logs"),
            heldout_eval_dir / "logs",
            heldout_root.parent / "logs",
            PROJECT_ROOT / "logs",
        ]
    )
    log_path = _resolve_path(args.log_file) if args.log_file else _resolve_latest_log(log_dirs)

    reports_dirs = _dedupe_paths(
        [
            _resolve_path(args.reports_dir)
            if args.reports_dir
            else _env_path("HELDOUT_REPORTS_DIR", heldout_eval_dir / "reports"),
            heldout_eval_dir / "reports",
            heldout_root.parent / "reports",
        ]
    )
    crashes_dirs = _dedupe_paths(
        [
            _resolve_path(args.crashes_dir)
            if args.crashes_dir
            else _env_path("HELDOUT_CRASHES_DIR", heldout_eval_dir / "crashes"),
            heldout_eval_dir / "crashes",
            heldout_root.parent / "crashes",
        ]
    )

    crash_report = (
        _resolve_path(args.crash_report)
        if args.crash_report
        else _first_existing([path / "collector_crash_report.json" for path in crashes_dirs])
        or heldout_eval_dir / "crashes" / "collector_crash_report.json"
    )

    worker_status_dir = (
        _resolve_path(args.worker_status_dir)
        if args.worker_status_dir
        else _first_existing([path / "dataset_workers" for path in reports_dirs])
        or heldout_eval_dir / "reports" / "dataset_workers"
    )

    report: dict[str, Any] = {
        "heldout_root": str(heldout_root),
        "candidate_paths": {
            "log_dirs": [str(path) for path in log_dirs],
            "selected_log": str(log_path) if log_path is not None else None,
            "reports_dirs": [str(path) for path in reports_dirs],
            "selected_worker_status_dir": str(worker_status_dir),
            "crashes_dirs": [str(path) for path in crashes_dirs],
            "selected_crash_report": str(crash_report),
        },
        "log": _parse_log(log_path) if log_path is not None else {"exists": False, "reason": "no log found"},
        "load_reports": _collect_load_reports(reports_dirs),
        "process_logs": _collect_process_logs(log_dirs),
        "collector_crash_report": _summarize_crash_report(crash_report),
        "worker_status": _summarize_worker_status(worker_status_dir),
        "recent_fatal_reports": [
            item
            for crashes_dir in crashes_dirs
            for item in _collect_fatal_reports(crashes_dir / "fatal", limit=12)
        ][:24],
    }
    report["failure_attribution"] = _build_failure_attribution(report)

    _print_section("Candidate Paths", report["candidate_paths"])
    _print_section("Failure Attribution", report["failure_attribution"])
    _print_section("Log Summary", report["log"])
    _print_section("Load Reports", report["load_reports"])
    _print_section("Process Logs", report["process_logs"])
    _print_section("Collector Crash Report", report["collector_crash_report"])
    _print_section("Worker Status", report["worker_status"])
    _print_section("Recent Fatal Reports", report["recent_fatal_reports"])

    if args.json_out:
        output_path = Path(args.json_out).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
