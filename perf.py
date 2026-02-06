from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Optional

import torch


def configure_torch(cfg):
    # TF32: big win on Ampere+ for matmuls/conv, keeps FP32 accumulation semantics.
    if torch.cuda.is_available() and bool(cfg.perf.tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.perf.cudnn_benchmark)


def unwrap_compiled(module):
    # torch.compile(nn.Module) returns an OptimizedModule with _orig_mod.
    return getattr(module, "_orig_mod", module)


def reset_parameters_(module):
    base = unwrap_compiled(module)

    def _reset(m):
        fn = getattr(m, "reset_parameters", None)
        if callable(fn):
            fn()

    base.apply(_reset)
    return module


def maybe_channels_last_(module, enabled: bool):
    if not enabled:
        return module
    base = unwrap_compiled(module)
    base.to(memory_format=torch.channels_last)
    return module


def maybe_compile(module, cfg):
    if not bool(cfg.perf.compile):
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    backend = str(getattr(cfg.perf, "compile_backend", "inductor"))
    return compile_fn(module, backend=backend)


def autocast_context(cfg, device: torch.device) -> ContextManager:
    amp = str(cfg.perf.amp).lower()
    if amp in ("off", "false", "0", "no"):
        return nullcontext()

    device = torch.device(device)
    if device.type == "cuda":
        if amp == "bf16":
            dtype = torch.bfloat16
        elif amp == "fp16":
            dtype = torch.float16
        elif amp == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise ValueError(f"Unknown perf.amp={cfg.perf.amp!r}")
        return torch.autocast(device_type="cuda", dtype=dtype)

    if device.type == "cpu":
        if amp in ("bf16", "auto"):
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    return nullcontext()


def make_grad_scaler(cfg, device: torch.device) -> Optional[torch.cuda.amp.GradScaler]:
    amp = str(cfg.perf.amp).lower()
    device = torch.device(device)
    if device.type != "cuda":
        return None
    if amp == "fp16" or (amp == "auto" and not torch.cuda.is_bf16_supported()):
        return torch.cuda.amp.GradScaler()
    return None

