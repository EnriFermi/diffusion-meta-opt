from __future__ import annotations

import contextlib
import logging
import random
import socket

import torch
from omegaconf import DictConfig


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
    if not bool(cfg[section].get(compile_key, False)):
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this PyTorch version; continuing without compile")
        return model

    compile_mode = str(cfg[section].get(compile_mode_key, "max-autotune"))
    logger.info("Compiling %s with torch.compile(mode=%s)", label, compile_mode)
    dynamic = bool(cfg[section].get("compile_dynamic", True))
    return torch.compile(model, mode=compile_mode, dynamic=dynamic)


def resolve_amp(cfg: DictConfig, device: torch.device, *, section: str, amp_key: str = "amp") -> tuple[bool, torch.dtype | None]:
    mode = str(cfg[section].get(amp_key, "auto")).lower()
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
