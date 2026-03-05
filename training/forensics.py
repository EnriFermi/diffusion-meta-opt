from __future__ import annotations

import errno
import json
import logging
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Full
from typing import Any

_STDIO_HANDLES: list[Any] = []
_DEFAULT_ENV_KEYS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "NCCL_ASYNC_ERROR_HANDLING",
    "PYTHONFAULTHANDLER",
)


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf  # type: ignore

        payload = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _safe_role(text: str) -> str:
    raw = str(text).strip().replace("/", "__").replace(" ", "_")
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw)
    return cleaned or "unknown"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_proc_status_fields(pid: int) -> dict[str, str]:
    text = _read_text(Path(f"/proc/{int(pid)}/status"))
    if not text:
        return {}
    payload: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _parse_kb_to_mb(raw_value: str | None) -> float | None:
    if not raw_value:
        return None
    token = str(raw_value).strip().split(" ", 1)[0]
    if not token:
        return None
    try:
        return float(int(token) / 1024.0)
    except Exception:
        return None


def _list_child_pids(pid: int) -> list[int]:
    text = _read_text(Path(f"/proc/{int(pid)}/task/{int(pid)}/children"))
    if not text:
        return []
    out: list[int] = []
    for token in text.strip().split():
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def collect_process_snapshot(pid: int, *, max_children: int = 8) -> dict[str, Any]:
    payload: dict[str, Any] = {"pid": int(pid)}
    fields = _read_proc_status_fields(pid)
    if fields:
        payload["name"] = fields.get("Name")
        payload["state"] = fields.get("State")
        payload["threads"] = int(fields.get("Threads", "0") or 0)
        payload["rss_mb"] = _parse_kb_to_mb(fields.get("VmRSS"))
        payload["hwm_mb"] = _parse_kb_to_mb(fields.get("VmHWM"))
        payload["vms_mb"] = _parse_kb_to_mb(fields.get("VmSize"))
        payload["swap_mb"] = _parse_kb_to_mb(fields.get("VmSwap"))

    fd_dir = Path(f"/proc/{int(pid)}/fd")
    try:
        payload["open_fds"] = len(list(fd_dir.iterdir()))
    except Exception:
        payload["open_fds"] = None

    children = _list_child_pids(pid)
    payload["children_count"] = len(children)
    children_rows: list[dict[str, Any]] = []
    children_rss_mb = 0.0
    for child_pid in children[: max(0, int(max_children))]:
        child = _read_proc_status_fields(child_pid)
        child_rss = _parse_kb_to_mb(child.get("VmRSS")) if child else None
        if child_rss is not None:
            children_rss_mb += float(child_rss)
        children_rows.append(
            {
                "pid": int(child_pid),
                "name": child.get("Name") if child else None,
                "state": child.get("State") if child else None,
                "rss_mb": child_rss,
                "threads": int(child.get("Threads", "0") or 0) if child else None,
            }
        )
    payload["children_rss_mb_sum"] = float(children_rss_mb)
    if children_rows:
        payload["children_head"] = children_rows
    return payload


def read_cgroup_memory_events() -> dict[str, int]:
    candidates = (
        Path("/sys/fs/cgroup/memory.events"),
        Path("/sys/fs/cgroup/memory/memory.events"),
    )
    for path in candidates:
        text = _read_text(path)
        if not text:
            continue
        out: dict[str, int] = {}
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            key, raw_value = parts
            try:
                out[str(key)] = int(raw_value)
            except Exception:
                continue
        if out:
            return out
    return {}


def _read_cgroup_scalar(paths: tuple[str, ...]) -> int | str | None:
    for path_text in paths:
        text = _read_text(Path(path_text))
        if not text:
            continue
        token = text.strip().splitlines()[0].strip()
        if not token:
            continue
        if token == "max":
            return token
        try:
            return int(token)
        except Exception:
            continue
    return None


def collect_cgroup_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "events": read_cgroup_memory_events(),
        "memory_current_bytes": _read_cgroup_scalar(
            ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
        ),
        "memory_max_bytes": _read_cgroup_scalar(
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
        ),
        "memory_swap_current_bytes": _read_cgroup_scalar(
            ("/sys/fs/cgroup/memory.swap.current", "/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes")
        ),
        "memory_swap_max_bytes": _read_cgroup_scalar(
            ("/sys/fs/cgroup/memory.swap.max", "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes")
        ),
    }
    return {k: v for k, v in payload.items() if v not in (None, {}, "")}


def collect_nvidia_snapshot() -> dict[str, Any]:
    commands = {
        "gpus": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    payload: dict[str, Any] = {}
    for key, cmd in commands.items():
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            payload[key] = {
                "returncode": int(completed.returncode),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except FileNotFoundError:
            payload[key] = {"error": "nvidia-smi not found"}
        except Exception as exc:
            payload[key] = {"error": str(exc)}
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_artifact_paths(cfg: Any) -> dict[str, Path]:
    cfg_dict = _to_plain_dict(cfg)
    artifacts = cfg_dict.get("training_artifacts") if isinstance(cfg_dict.get("training_artifacts"), dict) else {}

    root_dir = Path(str(artifacts.get("root_dir", "./artifacts/training")))
    logs_dir = Path(str(artifacts.get("logs_dir", root_dir / "logs")))
    reports_dir = Path(str(artifacts.get("reports_dir", root_dir / "reports")))
    crashes_dir = Path(str(artifacts.get("crashes_dir", root_dir / "crashes")))

    return {
        "root_dir": root_dir,
        "logs_dir": logs_dir,
        "reports_dir": reports_dir,
        "crashes_dir": crashes_dir,
    }


def _forensics_cfg(cfg: Any, *, section: str = "train") -> dict[str, Any]:
    cfg_dict = _to_plain_dict(cfg)
    section_cfg = cfg_dict.get(section) if isinstance(cfg_dict.get(section), dict) else {}
    forensics = section_cfg.get("forensics") if isinstance(section_cfg.get("forensics"), dict) else {}
    return forensics


def maybe_redirect_stdio(
    cfg: Any,
    *,
    role: str,
    section: str = "train",
    rank: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str] | None:
    forensics = _forensics_cfg(cfg, section=section)
    if not bool(forensics.get("enabled", False)):
        return None
    if not bool(forensics.get("stdio_redirect", True)):
        return None
    if os.environ.get("FORENSICS_STDIO_REDIRECTED_PID") == str(os.getpid()):
        return None

    paths = resolve_artifact_paths(cfg)
    stdio_dir = Path(str(forensics.get("stdio_dir", paths["logs_dir"] / "stdio")))
    stdio_dir.mkdir(parents=True, exist_ok=True)

    safe_role = _safe_role(role)
    rank_tag = f"_rank{rank}" if rank is not None else ""
    pid = int(os.getpid())
    stdout_path = stdio_dir / f"{safe_role}{rank_tag}_pid{pid}.stdout.log"
    stderr_path = stdio_dir / f"{safe_role}{rank_tag}_pid{pid}.stderr.log"

    stdout_handle = stdout_path.open("a", encoding="utf-8", buffering=1)
    stderr_handle = stderr_path.open("a", encoding="utf-8", buffering=1)
    _STDIO_HANDLES.extend([stdout_handle, stderr_handle])

    os.dup2(stdout_handle.fileno(), 1)
    os.dup2(stderr_handle.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", closefd=False)
    os.environ["FORENSICS_STDIO_REDIRECTED_PID"] = str(pid)
    os.environ["FORENSICS_STDOUT_PATH"] = str(stdout_path)
    os.environ["FORENSICS_STDERR_PATH"] = str(stderr_path)

    if logger is not None:
        logger.info("Forensics stdio redirect: stdout=%s stderr=%s", stdout_path, stderr_path)

    return {
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def maybe_enable_core_dumps(
    cfg: Any,
    *,
    section: str = "train",
    logger: logging.Logger | None = None,
) -> bool:
    forensics = _forensics_cfg(cfg, section=section)
    if not bool(forensics.get("enabled", False)):
        return False
    if not bool(forensics.get("enable_core_dumps", True)):
        return False
    try:
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_CORE)
        target_soft = resource.RLIM_INFINITY
        target_hard = current_hard
        if current_hard != resource.RLIM_INFINITY:
            target_hard = current_hard
            target_soft = min(current_hard, resource.RLIM_INFINITY)
        resource.setrlimit(resource.RLIMIT_CORE, (target_soft, target_hard))
        if logger is not None:
            logger.info(
                "Forensics core dumps: RLIMIT_CORE soft=%s hard=%s (previous soft=%s hard=%s)",
                target_soft,
                target_hard,
                current_soft,
                current_hard,
            )
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to enable core dumps (RLIMIT_CORE): %s", exc)
        return False


def apply_nccl_forensics_env(
    cfg: Any,
    *,
    section: str = "train",
    logger: logging.Logger | None = None,
) -> None:
    forensics = _forensics_cfg(cfg, section=section)
    if not bool(forensics.get("enabled", False)):
        return
    nccl_cfg = forensics.get("nccl") if isinstance(forensics.get("nccl"), dict) else {}
    if not bool(nccl_cfg.get("enabled", True)):
        return

    env_updates = {
        "TORCH_NCCL_TRACE_BUFFER_SIZE": str(int(nccl_cfg.get("trace_buffer_size", 1048576))),
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1" if bool(nccl_cfg.get("dump_on_timeout", True)) else "0",
        "NCCL_ASYNC_ERROR_HANDLING": "1" if bool(nccl_cfg.get("async_error_handling", True)) else "0",
        "PYTHONFAULTHANDLER": "1",
    }
    for key, value in env_updates.items():
        if key in os.environ and str(os.environ.get(key, "")).strip():
            continue
        os.environ[key] = value

    if logger is not None:
        logger.info(
            "Forensics NCCL env: trace_buffer=%s dump_on_timeout=%s async_error_handling=%s",
            os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE"),
            os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT"),
            os.environ.get("NCCL_ASYNC_ERROR_HANDLING"),
        )


def emit_fatal_report(
    cfg: Any,
    *,
    role: str,
    error: str,
    traceback_text: str,
    extra: dict[str, Any] | None = None,
    section: str = "train",
) -> str:
    paths = resolve_artifact_paths(cfg)
    forensics = _forensics_cfg(cfg, section=section)
    fatal_dir = Path(str(forensics.get("fatal_reports_dir", paths["crashes_dir"] / "fatal")))
    fatal_dir.mkdir(parents=True, exist_ok=True)

    timestamp = float(time.time())
    ts_ms = int(timestamp * 1000)
    safe_role = _safe_role(role)
    pid = int(os.getpid())
    path = fatal_dir / f"{safe_role}_pid{pid}_{ts_ms}.json"

    env_payload = {}
    for key in _DEFAULT_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        env_payload[key] = value

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "role": str(role),
        "pid": pid,
        "ppid": int(os.getppid()),
        "error": str(error),
        "traceback": str(traceback_text),
        "env": env_payload,
        "process_snapshot": collect_process_snapshot(pid),
        "cgroup": collect_cgroup_snapshot(),
        "gpu": collect_nvidia_snapshot(),
    }
    if extra:
        payload["extra"] = extra

    _write_json_atomic(path, payload)
    return str(path)


def monitor_send_event(queue_obj: Any | None, event: dict[str, Any]) -> bool:
    if queue_obj is None:
        return False
    payload = dict(event)
    payload.setdefault("timestamp", float(time.time()))
    try:
        queue_obj.put_nowait(payload)
        return True
    except Full:
        return False
    except Exception:
        return False


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    except Exception:
        return False


def _monitor_main(
    cfg_dict: dict[str, Any],
    section: str,
    label: str,
    queue_obj: Any,
    stop_event: Any,
) -> None:
    paths = resolve_artifact_paths(cfg_dict)
    forensics = _forensics_cfg(cfg_dict, section=section)
    monitor_dir = Path(str(forensics.get("monitor_dir", paths["reports_dir"] / "monitor")))
    monitor_dir.mkdir(parents=True, exist_ok=True)
    crashes_dir = Path(str(forensics.get("monitor_crashes_dir", paths["crashes_dir"] / "monitor")))
    crashes_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = monitor_dir / f"{_safe_role(label)}_events.jsonl"
    summary_path = monitor_dir / f"{_safe_role(label)}_summary.json"

    poll_s = max(0.1, float(forensics.get("monitor_poll_s", 1.0)))
    max_children = max(0, int(forensics.get("monitor_max_children", 12)))

    tracked: dict[int, dict[str, Any]] = {}
    dead_reports: dict[int, str] = {}

    def _record_event(event: dict[str, Any]) -> None:
        append_jsonl(event_log_path, event)

    def _mark_tracked(pid: int, *, role: str, event_type: str, metadata: dict[str, Any] | None = None) -> None:
        row = tracked.get(pid, {})
        row["pid"] = int(pid)
        row["role"] = str(role)
        row["last_event_type"] = str(event_type)
        row["last_seen_ts"] = float(time.time())
        if metadata:
            row["metadata"] = dict(metadata)
        tracked[pid] = row

    while True:
        now = float(time.time())
        drained = 0
        while drained < 512:
            try:
                event = queue_obj.get_nowait()
            except Empty:
                break
            except Exception:
                break

            drained += 1
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "unknown"))
            role = str(event.get("role", "unknown"))
            pid = event.get("pid")
            pid_int = int(pid) if isinstance(pid, int) else None

            _record_event(event)
            if pid_int is not None and pid_int > 0:
                _mark_tracked(pid_int, role=role, event_type=event_type, metadata=event.get("metadata"))

            if event_type == "fatal":
                fatal_payload = {
                    "timestamp": event.get("timestamp", now),
                    "event_type": event_type,
                    "role": role,
                    "pid": pid_int,
                    "error": event.get("error"),
                    "traceback": event.get("traceback"),
                    "metadata": event.get("metadata"),
                    "monitor_snapshot": collect_process_snapshot(os.getpid(), max_children=max_children),
                    "cgroup": collect_cgroup_snapshot(),
                    "gpu": collect_nvidia_snapshot(),
                }
                fatal_path = crashes_dir / (
                    f"{_safe_role(role)}_pid{pid_int or 0}_{int(float(event.get('timestamp', now)) * 1000)}.json"
                )
                _write_json_atomic(fatal_path, fatal_payload)
                if pid_int is not None:
                    dead_reports[pid_int] = str(fatal_path)

            children = event.get("tracked_children")
            if isinstance(children, list):
                for child in children:
                    if not isinstance(child, dict):
                        continue
                    child_pid = child.get("pid")
                    if not isinstance(child_pid, int) or child_pid <= 0:
                        continue
                    child_role = str(child.get("role", "child"))
                    child_meta = child.get("metadata") if isinstance(child.get("metadata"), dict) else {}
                    _mark_tracked(child_pid, role=child_role, event_type="tracked_child", metadata=child_meta)

            if event_type == "exit" and pid_int is not None:
                tracked[pid_int]["exited"] = True
                tracked[pid_int]["exit_code"] = event.get("exit_code")
                tracked[pid_int]["exit_ts"] = float(event.get("timestamp", now))

        for pid, row in list(tracked.items()):
            if row.get("exited") and pid in dead_reports:
                continue
            if row.get("exited"):
                dead_reports[pid] = "graceful_exit"
                continue
            if _pid_exists(pid):
                continue
            if pid in dead_reports:
                continue

            death_payload = {
                "timestamp": now,
                "event_type": "observed_process_death",
                "pid": int(pid),
                "role": row.get("role"),
                "last_seen_ts": row.get("last_seen_ts"),
                "last_event_type": row.get("last_event_type"),
                "metadata": row.get("metadata"),
                "monitor_snapshot": collect_process_snapshot(os.getpid(), max_children=max_children),
                "cgroup": collect_cgroup_snapshot(),
                "gpu": collect_nvidia_snapshot(),
            }
            death_path = crashes_dir / f"{_safe_role(str(row.get('role', 'process')))}_pid{pid}_{int(now * 1000)}.json"
            _write_json_atomic(death_path, death_payload)
            dead_reports[pid] = str(death_path)
            _record_event(
                {
                    "timestamp": now,
                    "type": "process_death_report_written",
                    "pid": int(pid),
                    "role": row.get("role"),
                    "path": str(death_path),
                }
            )

        if stop_event.is_set():
            try:
                pending = queue_obj.qsize()
            except Exception:
                pending = 0
            if pending <= 0:
                break
        time.sleep(poll_s)

    summary_payload = {
        "timestamp": float(time.time()),
        "label": str(label),
        "tracked_count": len(tracked),
        "tracked_pids": sorted(int(pid) for pid in tracked.keys()),
        "dead_reports": dead_reports,
        "event_log_path": str(event_log_path),
    }
    _write_json_atomic(summary_path, summary_payload)


def start_process_monitor(
    cfg: Any,
    *,
    section: str = "train",
    label: str = "training",
) -> tuple[mp.Process | None, Any | None, Any | None]:
    forensics = _forensics_cfg(cfg, section=section)
    if not bool(forensics.get("enabled", False)):
        return None, None, None
    if not bool(forensics.get("monitor_enabled", True)):
        return None, None, None

    ctx = mp.get_context("spawn")
    queue_max_items = max(128, int(forensics.get("monitor_queue_max_items", 8192)))
    queue_obj = ctx.Queue(maxsize=queue_max_items)
    stop_event = ctx.Event()
    cfg_dict = _to_plain_dict(cfg)

    process = ctx.Process(
        target=_monitor_main,
        args=(cfg_dict, str(section), str(label), queue_obj, stop_event),
        daemon=False,
        name=f"forensics_monitor_{_safe_role(label)}",
    )
    process.start()
    return process, queue_obj, stop_event


def stop_process_monitor(
    process: mp.Process | None,
    queue_obj: Any | None,
    stop_event: Any | None,
    *,
    timeout_s: float = 10.0,
) -> None:
    if stop_event is not None:
        try:
            stop_event.set()
        except Exception:
            pass
    if process is not None:
        process.join(timeout=max(0.1, float(timeout_s)))
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
    if queue_obj is not None:
        try:
            queue_obj.close()
        except Exception:
            pass
