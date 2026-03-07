from __future__ import annotations

import contextlib
import datetime as dt
import inspect
import logging
import os
import random
import socket
import uuid
from pathlib import Path

import torch
from omegaconf import DictConfig, open_dict


def get_rank_logger(name: str, rank: int) -> logging.Logger:
    return logging.getLogger(f"{name}.rank{rank}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_world_size(
    cfg: DictConfig,
    *,
    section: str,
    distributed_key: str = "distributed",
    num_gpus_key: str = "num_gpus",
    error_prefix: str,
) -> int:
    train_cfg = cfg[section]
    distributed_mode = str(train_cfg.get(distributed_key, "auto")).strip().lower()
    num_gpus_req = int(train_cfg.get(num_gpus_key, 0))

    if distributed_mode in {"false", "0", "off", "no"}:
        return 1

    if not torch.cuda.is_available():
        return 1

    num_available = int(torch.cuda.device_count())
    if num_available <= 1:
        return 1

    if num_gpus_req > 0:
        num_available = min(num_available, num_gpus_req)

    if distributed_mode in {"true", "1", "on", "yes"} and num_available < 2:
        raise ValueError(f"{error_prefix}.{distributed_key}=true requires at least 2 visible CUDA devices")

    return max(1, num_available)


def resolve_backend(cfg: DictConfig, device: torch.device, *, section: str, backend_key: str = "backend") -> str:
    raw = str(cfg[section].get(backend_key, "auto")).lower()
    if raw != "auto":
        return raw
    if device.type == "cuda":
        return "nccl"
    return "gloo"


def resolve_device(
    cfg: DictConfig,
    rank: int,
    world_size: int,
    *,
    section: str,
    device_key: str = "device",
    default_single_gpu_device: str = "cuda:0",
) -> torch.device:
    if torch.cuda.is_available():
        if world_size > 1:
            dev = torch.device(f"cuda:{rank}")
        else:
            wanted = str(cfg[section].get(device_key, default_single_gpu_device))
            dev = torch.device(wanted if wanted.startswith("cuda") else default_single_gpu_device)
        torch.cuda.set_device(dev)
        return dev

    return torch.device("cpu")


def set_speed_optimizations(
    cfg: DictConfig,
    device: torch.device,
    *,
    section: str,
    tf32_key: str = "tf32",
    cudnn_benchmark_key: str = "cudnn_benchmark",
) -> None:
    section_cfg = cfg[section]
    tf32 = bool(section_cfg.get(tf32_key, True))
    cudnn_benchmark = bool(section_cfg.get(cudnn_benchmark_key, True))

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark

        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def maybe_compile_model(
    model: torch.nn.Module,
    cfg: DictConfig,
    logger: logging.Logger,
    *,
    section: str,
    compile_key: str = "compile",
    compile_mode_key: str = "compile_mode",
    label: str = "model",
) -> torch.nn.Module:
    compile_enabled = bool(cfg[section].get(compile_key, False))
    if not compile_enabled:
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this PyTorch version; continuing without compile")
        return model

    compile_mode = str(cfg[section].get(compile_mode_key, "max-autotune"))
    dynamic = bool(cfg[section].get("compile_dynamic", True))
    backend = cfg[section].get("compile_backend")
    disable_cudagraphs = bool(cfg[section].get("compile_disable_cudagraphs", False))
    backend_name = str(backend).strip() if backend is not None else ""

    logger.info(
        "Compiling %s with torch.compile(mode=%s, dynamic=%s%s)",
        label,
        compile_mode,
        dynamic,
        f", backend={backend_name}" if backend_name else "",
    )

    compile_kwargs: dict[str, object] = {"dynamic": dynamic}
    if backend_name:
        compile_kwargs["backend"] = backend_name

    compile_options_raw = cfg[section].get("compile_options")
    compile_options: dict[str, object] = {}
    if isinstance(compile_options_raw, (dict, DictConfig)):
        for key, value in compile_options_raw.items():
            compile_options[str(key)] = value
    if disable_cudagraphs:
        # Prevent output-buffer reuse issues from inductor cudagraphs on dynamic workloads.
        compile_options.setdefault("triton.cudagraphs", False)
        logger.info("Disabling inductor cudagraphs for %s", label)

    compile_signature = inspect.signature(torch.compile).parameters
    if compile_options:
        if "options" in compile_signature:
            compile_kwargs["options"] = compile_options
            logger.info(
                "Using torch.compile options for %s; omitting mode=%s because mode and options are mutually exclusive",
                label,
                compile_mode,
            )
        else:
            logger.warning(
                "torch.compile(options=...) unsupported in this PyTorch version; "
                "compile_options for %s will be ignored",
                label,
            )
            if disable_cudagraphs:
                try:
                    # Backward-compat fallback for older torch versions.
                    import torch._inductor.config as inductor_config

                    triton_cfg = getattr(inductor_config, "triton", None)
                    if triton_cfg is not None and hasattr(triton_cfg, "cudagraphs"):
                        setattr(triton_cfg, "cudagraphs", False)
                    elif hasattr(inductor_config, "cudagraphs"):
                        setattr(inductor_config, "cudagraphs", False)
                    else:
                        logger.warning(
                            "Requested compile_disable_cudagraphs for %s but could not find "
                            "a known torch._inductor.config cudagraph switch",
                            label,
                        )
                except Exception as exc:
                    logger.warning("Failed to disable inductor cudagraphs for %s: %s", label, exc)
            compile_kwargs["mode"] = compile_mode
    else:
        compile_kwargs["mode"] = compile_mode
    if "mode" in compile_kwargs and compile_mode == "max-autotune" and dynamic:
        logger.warning(
            "%s.%s=max-autotune with %s.compile_dynamic=true can be unstable on some CUDA stacks; "
            "prefer mode=reduce-overhead and compile_dynamic=false for stability",
            section,
            compile_mode_key,
            section,
        )
    return torch.compile(model, **compile_kwargs)


def resolve_amp(cfg: DictConfig, device: torch.device, *, section: str, amp_key: str = "amp") -> tuple[bool, torch.dtype | None]:
    mode_raw = cfg[section].get(amp_key, "auto")
    if isinstance(mode_raw, bool):
        mode = "auto" if mode_raw else "off"
    else:
        mode = str(mode_raw).lower()

    if mode in {"on", "true", "1", "yes"}:
        mode = "auto"

    if device.type != "cuda" or mode in {"off", "false", "0", "none"}:
        return False, None

    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16

    if mode in {"bf16", "bfloat16"}:
        return True, torch.bfloat16

    if mode in {"fp16", "float16", "half"}:
        return True, torch.float16

    raise ValueError(f"Unsupported {section}.{amp_key}={mode}")


def autocast_context(enabled: bool, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _safe_run_label(text: str) -> str:
    raw = str(text).strip().replace(" ", "_").replace("/", "__")
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw)
    return cleaned or "run"


def configure_per_run_artifacts(
    cfg: DictConfig,
    *,
    run_label: str,
) -> dict[str, str]:
    """
    Configure per-run artifact directories under training_artifacts.

    By default this creates:
      <base_root_dir>/runs/<run_id>/{logs,reports,crashes,checkpoints/...}
    """

    with open_dict(cfg):
        if not isinstance(cfg.get("training_artifacts"), (dict, DictConfig)):
            cfg["training_artifacts"] = {}
        ta = cfg["training_artifacts"]

        base_root = Path(str(ta.get("base_root_dir", ta.get("root_dir", "./artifacts/training"))))
        separate_run_dirs = bool(ta.get("separate_run_dirs", True))
        runs_dir = Path(str(ta.get("runs_dir", base_root / "runs")))

        run_id = str(ta.get("run_id", "")).strip()
        if not run_id:
            run_id = str(os.environ.get("TRAINING_RUN_ID", "")).strip()
        if not run_id:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{_safe_run_label(run_label)}_{timestamp}_pid{os.getpid()}_{uuid.uuid4().hex[:6]}"

        root_dir = (runs_dir / run_id) if separate_run_dirs else base_root
        logs_dir = root_dir / "logs"
        reports_dir = root_dir / "reports"
        crashes_dir = root_dir / "crashes"
        mini_ckpt_dir = root_dir / "checkpoints" / "mini_patch_vae"
        big_ckpt_dir = root_dir / "checkpoints" / "weight_quantile_vae"

        ta["base_root_dir"] = str(base_root)
        ta["separate_run_dirs"] = bool(separate_run_dirs)
        ta["runs_dir"] = str(runs_dir)
        ta["run_id"] = run_id
        ta["root_dir"] = str(root_dir)
        ta["logs_dir"] = str(logs_dir)
        ta["reports_dir"] = str(reports_dir)
        ta["crashes_dir"] = str(crashes_dir)
        ta["mini_vae_checkpoint_dir"] = str(mini_ckpt_dir)
        ta["big_vae_checkpoint_dir"] = str(big_ckpt_dir)
        ta["mini_encoder_latest_checkpoint"] = str(mini_ckpt_dir / "mini_encoder_latest.pt")

        if isinstance(cfg.get("logging"), (dict, DictConfig)):
            cfg["logging"]["dir"] = str(logs_dir)

        if isinstance(cfg.get("train"), (dict, DictConfig)) and str(cfg["train"].get("checkpoint_dir", "")).strip():
            cfg["train"]["checkpoint_dir"] = str(big_ckpt_dir)

        if isinstance(cfg.get("mini_train"), (dict, DictConfig)) and str(cfg["mini_train"].get("checkpoint_dir", "")).strip():
            cfg["mini_train"]["checkpoint_dir"] = str(mini_ckpt_dir)

    for path in (root_dir, logs_dir, reports_dir, crashes_dir, mini_ckpt_dir, big_ckpt_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_id,
        "root_dir": str(root_dir),
        "logs_dir": str(logs_dir),
        "reports_dir": str(reports_dir),
        "crashes_dir": str(crashes_dir),
        "mini_vae_checkpoint_dir": str(mini_ckpt_dir),
        "big_vae_checkpoint_dir": str(big_ckpt_dir),
    }
