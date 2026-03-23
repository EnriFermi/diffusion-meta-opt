from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import logging
import math
import os
import random
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler

from models.big_weight_vae import BigVAEConfig, BigWeightVAE, EncoderConfig, ModelConfig, TTMMemoryConfig
from models.distribution_encoder import DistributionConfig
from models.mini_patch_vae import MiniVAEConfig
from models.vae_shared import _quantile_over_samples
from procedural_simple_vae_pipeline.model import build_procedural_simple_vae


def _logger(name: str, rank: int) -> logging.Logger:
    return logging.getLogger(f"{name}.rank{rank}")


def _safe_run_label(text: str) -> str:
    raw = str(text).strip().replace(" ", "_").replace("/", "__")
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw)
    return cleaned or "run"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = [chunk for chunk in dotted_key.split(".") if chunk]
    if not parts:
        raise ValueError("Override key cannot be empty")
    for chunk in parts[:-1]:
        next_value = current.get(chunk)
        if not isinstance(next_value, dict):
            next_value = {}
            current[chunk] = next_value
        current = next_value
    current[parts[-1]] = value


def _parse_override_value(raw_value: str) -> Any:
    parsed = yaml.safe_load(raw_value)
    return parsed


def _load_config(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Top-level config must be a mapping, got {type(cfg)}")

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have form key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        _set_nested(cfg, key.strip(), _parse_override_value(raw_value))
    return cfg


def _configure_run_artifacts(cfg: dict[str, Any]) -> dict[str, str]:
    artifacts = cfg.setdefault("training_artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("training_artifacts must be a mapping")

    base_root = Path(str(artifacts.get("base_root_dir", "./artifacts/procedural_simple_vae")))
    separate_run_dirs = bool(artifacts.get("separate_run_dirs", True))
    runs_dir = Path(str(artifacts.get("runs_dir", base_root / "runs")))

    run_id = str(artifacts.get("run_id", "")).strip()
    if not run_id:
        run_id = str(os.environ.get("TRAINING_RUN_ID", "")).strip()
    if not run_id:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"procedural_simple_vae_train_{timestamp}_pid{os.getpid()}_{uuid.uuid4().hex[:6]}"

    root_dir = (runs_dir / run_id) if separate_run_dirs else base_root
    logs_dir = root_dir / "logs"
    reports_dir = root_dir / "reports"
    crashes_dir = root_dir / "crashes"
    checkpoint_dir = root_dir / "checkpoints" / "procedural_simple_vae"

    for path in (root_dir, logs_dir, reports_dir, crashes_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    logging_cfg = cfg.setdefault("logging", {})
    if not isinstance(logging_cfg, dict):
        raise TypeError("logging must be a mapping")
    file_name = str(logging_cfg.get("file_name", "")).strip()
    if not file_name:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_name = str(logging_cfg.get("project_name", "procedural_simple_vae")).strip() or "procedural_simple_vae"
        file_name = f"{project_name}_procedural_simple_vae_train_{timestamp}.log"
    file_path = logging_cfg.get("file_path")
    if file_path:
        resolved_log_path = Path(str(file_path))
    else:
        resolved_log_path = logs_dir / file_name

    train_cfg = cfg.setdefault("train", {})
    if not isinstance(train_cfg, dict):
        raise TypeError("train must be a mapping")

    artifacts["base_root_dir"] = str(base_root)
    artifacts["separate_run_dirs"] = separate_run_dirs
    artifacts["runs_dir"] = str(runs_dir)
    artifacts["run_id"] = run_id
    artifacts["root_dir"] = str(root_dir)
    artifacts["logs_dir"] = str(logs_dir)
    artifacts["reports_dir"] = str(reports_dir)
    artifacts["crashes_dir"] = str(crashes_dir)
    artifacts["procedural_simple_vae_checkpoint_dir"] = str(checkpoint_dir)

    logging_cfg["dir"] = str(logs_dir)
    logging_cfg["file_name"] = file_name
    logging_cfg["file_path"] = str(resolved_log_path)
    train_cfg["checkpoint_dir"] = str(checkpoint_dir)

    return {
        "run_id": run_id,
        "root_dir": str(root_dir),
        "logs_dir": str(logs_dir),
        "reports_dir": str(reports_dir),
        "crashes_dir": str(crashes_dir),
        "procedural_simple_vae_checkpoint_dir": str(checkpoint_dir),
    }


def _setup_logging(cfg: dict[str, Any], rank: int = 0) -> None:
    data_cfg = cfg.get("data", {})
    logging_cfg = cfg.get("logging", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}

    level_name = str(data_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    if rank != 0:
        level = max(level, logging.WARNING)

    fmt = str(logging_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log_path = Path(str(logging_cfg.get("file_path", "./artifacts/procedural_simple_vae/run.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(filename=log_path, mode="a", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt))
    logging.basicConfig(level=level, handlers=handlers, force=True)


def _resolve_world_size(cfg: dict[str, Any]) -> int:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        return 1
    distributed_mode = str(train_cfg.get("distributed", "auto")).strip().lower()
    num_gpus_req = int(train_cfg.get("num_gpus", 0))

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
        raise ValueError("train.distributed=true requires at least 2 visible CUDA devices")

    return max(1, num_available)


def _resolve_device(cfg: dict[str, Any], rank: int, world_size: int) -> torch.device:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    if torch.cuda.is_available():
        if world_size > 1:
            device = torch.device(f"cuda:{rank}")
        else:
            wanted = str(train_cfg.get("device", "cuda:0"))
            device = torch.device(wanted if wanted.startswith("cuda") else "cuda:0")
        torch.cuda.set_device(device)
        return device

    return torch.device("cpu")


def _resolve_backend(cfg: dict[str, Any], device: torch.device) -> str:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    raw = str(train_cfg.get("backend", "auto")).lower()
    if raw != "auto":
        return raw
    return "nccl" if device.type == "cuda" else "gloo"


def _set_speed_optimizations(cfg: dict[str, Any], device: torch.device) -> None:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    tf32 = bool(train_cfg.get("tf32", True))
    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", True))

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


def _maybe_compile(model: torch.nn.Module, cfg: dict[str, Any], logger: logging.Logger) -> torch.nn.Module:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        return model
    if not bool(train_cfg.get("compile", False)):
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable; continuing without compile")
        return model

    compile_mode = str(train_cfg.get("compile_mode", "max-autotune"))
    dynamic = bool(train_cfg.get("compile_dynamic", True))
    logger.info("Compiling procedural_simple_vae with torch.compile(mode=%s, dynamic=%s)", compile_mode, dynamic)
    return torch.compile(model, mode=compile_mode, dynamic=dynamic)


def _resolve_amp(cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype | None]:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    mode_raw = train_cfg.get("amp", "auto")
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
    raise ValueError(f"Unsupported train.amp={mode}")


def _autocast_context(enabled: bool, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_optimizer(model: torch.nn.Module, cfg: dict[str, Any], device: torch.device) -> torch.optim.Optimizer:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    betas_cfg = train_cfg.get("betas", [0.9, 0.999])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])
    eps = float(train_cfg.get("eps", 1e-8))

    kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "eps": eps,
    }
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = device.type == "cuda"
    if "foreach" in torch.optim.AdamW.__init__.__code__.co_varnames and not kwargs.get("fused", False):
        kwargs["foreach"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]) -> torch.optim.lr_scheduler.LambdaLR:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    max_steps = max(1, int(train_cfg.get("max_steps", 1000)))
    warmup_steps = max(0, int(train_cfg.get("warmup_steps", 100)))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = (step_idx - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_model_cfg(cfg: dict[str, Any]) -> ModelConfig:
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise TypeError("model must be a mapping")
    dist_cfg = model_cfg.get("distribution", {})
    mini_cfg = model_cfg.get("mini_vae", {})
    big_cfg = model_cfg.get("big_vae", {})
    enc_cfg = big_cfg.get("encoder", {})
    ttm_cfg = big_cfg.get("ttm", {})
    if (
        not isinstance(dist_cfg, dict)
        or not isinstance(mini_cfg, dict)
        or not isinstance(big_cfg, dict)
        or not isinstance(enc_cfg, dict)
        or not isinstance(ttm_cfg, dict)
    ):
        raise TypeError("model sub-sections must be mappings")

    patch_size = int(model_cfg.get("patch_size", 16))
    return ModelConfig(
        patch_size=patch_size,
        beta=float(model_cfg.get("beta", 1e-3)),
        variant=str(model_cfg.get("variant", "procedural_simple")),
        distribution=DistributionConfig(
            k_s=int(dist_cfg.get("k_s", 16)),
            Kq=int(dist_cfg.get("Kq", 32)),
            d_var=int(dist_cfg.get("d_var", 128)),
            d_dist=int(dist_cfg.get("d_dist", 128)),
            num_var_attn_layers=int(dist_cfg.get("num_var_attn_layers", 2)),
            var_attn_heads=int(dist_cfg.get("var_attn_heads", 4)),
            dcn_num_cross_layers=int(dist_cfg.get("dcn_num_cross_layers", 3)),
            dcn_deep_hidden=int(dist_cfg.get("dcn_deep_hidden", 0)),
            dcn_deep_layers=int(dist_cfg.get("dcn_deep_layers", 0)),
            dropout=float(dist_cfg.get("dropout", 0.0)),
            use_covariance=bool(dist_cfg.get("use_covariance", True)),
            patch_size_for_cov=int(dist_cfg.get("patch_size_for_cov", patch_size)),
        ),
        mini_vae=MiniVAEConfig(
            z_dim=int(mini_cfg.get("z_dim", 64)),
            d_e=int(mini_cfg.get("d_e", 128)),
            encoder_latent_dim=int(mini_cfg.get("encoder_latent_dim", 0)),
            pos_lat_dim=int(mini_cfg.get("pos_lat_dim", 0)),
            pos_dim=int(mini_cfg.get("pos_dim", 32)),
            num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
            num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
            decoder_bilinear_rank=int(mini_cfg.get("decoder_bilinear_rank", 0)),
            decoder_L_latents=int(mini_cfg.get("decoder_L_latents", 8)),
            decoder_use_dist_conditioning=bool(mini_cfg.get("decoder_use_dist_conditioning", True)),
            decoder_dist_mode=str(mini_cfg.get("decoder_dist_mode", "add")),
            use_latent_sampling=bool(mini_cfg.get("use_latent_sampling", True)),
            implementation=str(mini_cfg.get("implementation", "real")),
            mlp_stub_hidden_dim=int(mini_cfg.get("mlp_stub_hidden_dim", 256)),
            stub_mlp_use_batchnorm=bool(mini_cfg.get("stub_mlp_use_batchnorm", False)),
            stub_resampler_d_model=int(mini_cfg.get("stub_resampler_d_model", 0)),
            init_style=str(mini_cfg.get("init_style", "llm")),
            n_heads=int(mini_cfg.get("n_heads", 4)),
            d_patch=int(mini_cfg.get("d_patch", 64)),
            dropout=float(mini_cfg.get("dropout", 0.0)),
        ),
        big_vae=BigVAEConfig(
            d_model=int(big_cfg.get("d_model", 256)),
            d_lat=int(big_cfg.get("d_lat", 256)),
            num_latents=int(big_cfg.get("num_latents", 32)),
            num_encoder_layers=int(big_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(big_cfg.get("num_decoder_layers", 4)),
            n_heads=int(big_cfg.get("n_heads", 8)),
            ffn_mult=float(big_cfg.get("ffn_mult", 4.0)),
            dropout=float(big_cfg.get("dropout", 0.0)),
            pos_fourier_dim=int(big_cfg.get("pos_fourier_dim", 64)),
            use_latent_sampling=bool(big_cfg.get("use_latent_sampling", True)),
            disable_z_shortcut=bool(big_cfg.get("disable_z_shortcut", False)),
            disable_distribution_encoder=bool(big_cfg.get("disable_distribution_encoder", False)),
            latent_bottleneck_kind=str(big_cfg.get("latent_bottleneck_kind", "ttm")),
            ttm=TTMMemoryConfig(
                proc_tokens=int(ttm_cfg.get("proc_tokens", 8)),
                process_depth=int(ttm_cfg.get("process_depth", 2)),
                summarizer_mode=str(ttm_cfg.get("summarizer_mode", "mlp")),
                summarizer_hidden_mult=float(ttm_cfg.get("summarizer_hidden_mult", 2.0)),
                num_blocks=int(ttm_cfg.get("num_blocks", 1)),
                share_weights=bool(ttm_cfg.get("share_weights", False)),
                dropout=float(ttm_cfg.get("dropout", big_cfg.get("dropout", 0.0))),
                use_type_embeddings=bool(ttm_cfg.get("use_type_embeddings", True)),
                use_positional_embeddings=bool(ttm_cfg.get("use_positional_embeddings", True)),
                memory_init=str(ttm_cfg.get("memory_init", "learned")),
                return_aux=bool(ttm_cfg.get("return_aux", False)),
            ),
            encoder=EncoderConfig(
                self_attn_mode=str(enc_cfg.get("self_attn_mode", "full")),
                cross_attend_only_cls=bool(enc_cfg.get("cross_attend_only_cls", True)),
            ),
        ),
    )


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: dict[str, Any],
    step_idx: int,
    logger: logging.Logger,
) -> None:
    train_cfg = cfg.get("train", {})
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "./artifacts/procedural_simple_vae/checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model
    payload = {
        "step": int(step_idx),
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": cfg,
    }

    step_path = checkpoint_dir / f"step_{int(step_idx):07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    logger.info("Checkpoint saved: %s", step_path)


def _load_model_weights_from_checkpoint(model: torch.nn.Module, path: str, logger: logging.Logger) -> None:
    logger.info("Loading model weights from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"]
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(state_dict, strict=True)
    logger.info("Model weights loaded successfully (step=%s)", ckpt.get("step", "?"))


def _build_patch_stats_teacher_residual(
    *,
    cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
    batch_size: int,
    n_rows: int,
    d_in: int,
    d_out: int,
    x_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise TypeError("model must be a mapping")
    patch_size = max(1, int(model_cfg.get("patch_size", 16)))
    if d_in % patch_size != 0:
        raise ValueError(
            "teacher synthetic modes currently require d_in to be divisible by model.patch_size, "
            f"got d_in={d_in} patch_size={patch_size}"
        )

    patch_mean_std = float(teacher_cfg.get("patch_mean_std", x_std))
    patch_scale_min = float(teacher_cfg.get("patch_scale_min", 0.5))
    patch_scale_max = float(teacher_cfg.get("patch_scale_max", 1.5))
    mixture_shift_std = float(teacher_cfg.get("mixture_shift_std", 0.75 * x_std))
    q_low = float(teacher_cfg.get("q_low", 0.15))
    q_high = float(teacher_cfg.get("q_high", 0.85))

    if patch_mean_std <= 0.0:
        raise ValueError(f"teacher.patch_mean_std must be > 0, got {patch_mean_std}")
    if patch_scale_min <= 0.0 or patch_scale_max < patch_scale_min:
        raise ValueError(
            "teacher patch scales must satisfy "
            f"0 < patch_scale_min <= patch_scale_max, got {patch_scale_min}, {patch_scale_max}"
        )
    if mixture_shift_std < 0.0:
        raise ValueError(f"teacher.mixture_shift_std must be >= 0, got {mixture_shift_std}")
    if not (0.0 <= q_low < 0.5 < q_high <= 1.0):
        raise ValueError(
            "teacher quantiles must satisfy 0 <= q_low < 0.5 < q_high <= 1, "
            f"got q_low={q_low} q_high={q_high}"
        )

    T = d_in // patch_size
    patch_means = torch.randn((batch_size, T, patch_size), device=device, dtype=torch.float32) * patch_mean_std
    raw_patch_scales = torch.randn((batch_size, T, patch_size), device=device, dtype=torch.float32)
    patch_scales = patch_scale_min + (patch_scale_max - patch_scale_min) * torch.sigmoid(raw_patch_scales)
    patch_mix_shifts = torch.randn((batch_size, T, patch_size), device=device, dtype=torch.float32) * mixture_shift_std

    gaussian_noise = torch.randn((batch_size, n_rows, T, patch_size), device=device, dtype=torch.float32)
    mixture_selector = torch.sign(torch.randn((batch_size, n_rows, T, 1), device=device, dtype=torch.float32))
    mixture_selector = torch.where(mixture_selector == 0, torch.ones_like(mixture_selector), mixture_selector)
    x_patches = (
        patch_means.unsqueeze(1)
        + x_std * patch_scales.unsqueeze(1) * gaussian_noise
        + patch_mix_shifts.unsqueeze(1) * mixture_selector
    )

    x_patch_flat = x_patches.permute(0, 2, 1, 3).reshape(batch_size * T, n_rows, patch_size)
    q_probs = torch.tensor([q_low, 0.5, q_high], device=device, dtype=torch.float32)
    q_values = _quantile_over_samples(x_patch_flat, q_probs)
    q_low_patch = q_values[0].view(batch_size, T, patch_size)
    q_mid_patch = q_values[1].view(batch_size, T, patch_size)
    q_high_patch = q_values[2].view(batch_size, T, patch_size)
    q_spread_patch = q_high_patch - q_low_patch
    q_center_patch = 0.5 * (q_high_patch + q_low_patch)
    patch_global_center = q_mid_patch.mean(dim=-1, keepdim=True)
    patch_global_spread = q_spread_patch.mean(dim=-1, keepdim=True)

    feat_idx = torch.arange(patch_size, device=device, dtype=torch.float32).view(1, 1, patch_size, 1)
    out_idx = torch.arange(d_out, device=device, dtype=torch.float32).view(1, 1, 1, d_out)
    basis_a = torch.cos(0.13 * (feat_idx + 1.0) * (out_idx + 1.0))
    basis_b = torch.sin(0.17 * (feat_idx + 1.0) * (out_idx + 1.0))
    basis_c = torch.cos(0.07 * (feat_idx + 1.0) + 0.19 * (out_idx + 1.0))
    basis_d = torch.sin(0.11 * (feat_idx + 1.0) + 0.23 * (out_idx + 1.0))

    residual_patches = (
        q_mid_patch.unsqueeze(-1) * basis_a
        + 0.7 * q_spread_patch.unsqueeze(-1) * basis_b
        + 0.35 * q_center_patch.unsqueeze(-1) * basis_c
        + 0.2 * patch_global_center.unsqueeze(-1) * basis_d
        + 0.15 * patch_global_spread.unsqueeze(-1) * basis_c
    ) / max(float(patch_size) ** 0.5, 1.0)

    x = x_patches.reshape(batch_size, n_rows, d_in)
    residual = residual_patches.reshape(batch_size, d_in, d_out)
    return x, residual


def _build_masked_ab_from_x_batch(
    *,
    synth_cfg: dict[str, Any],
    batch_size: int,
    n_rows: int,
    d_in: int,
    d_out: int,
    x_std: float,
    w_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_cfg = synth_cfg.get("masked_ab_from_x", {})
    if not isinstance(teacher_cfg, dict):
        raise TypeError("train.synthetic.masked_ab_from_x must be a mapping")
    if d_out % 2 != 0:
        raise ValueError(
            "train.synthetic.kind='masked_ab_from_x' requires d_out to be even, "
            f"got d_out={d_out}"
        )

    a_std = float(teacher_cfg.get("a_std", w_std))
    c_std = float(teacher_cfg.get("c_std", w_std))
    x_noise_std = float(teacher_cfg.get("x_noise_std", x_std))
    mask_value = float(teacher_cfg.get("mask_value", 0.0))
    if a_std < 0.0:
        raise ValueError(f"train.synthetic.masked_ab_from_x.a_std must be >= 0, got {a_std}")
    if c_std < 0.0:
        raise ValueError(f"train.synthetic.masked_ab_from_x.c_std must be >= 0, got {c_std}")
    if x_noise_std <= 0.0:
        raise ValueError(f"train.synthetic.masked_ab_from_x.x_noise_std must be > 0, got {x_noise_std}")

    half_out = d_out // 2
    A = a_std * torch.randn((batch_size, d_in, half_out), device=device, dtype=torch.float32)
    C = c_std * torch.randn((batch_size, d_in, 1), device=device, dtype=torch.float32)
    B = A + C

    W_target = torch.cat([A, B], dim=2)
    masked_half = torch.full_like(B, mask_value)
    W_input = torch.cat([A, masked_half], dim=2)

    x_mean = C.squeeze(-1)
    X = x_mean.unsqueeze(1) + x_noise_std * torch.randn((batch_size, n_rows, d_in), device=device, dtype=torch.float32)
    return X, W_input, W_target


def _sample_procedural_batch(cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cfg = cfg.get("train", {})
    synth_cfg = train_cfg.get("synthetic", {}) if isinstance(train_cfg, dict) else {}
    if not isinstance(synth_cfg, dict):
        raise TypeError("train.synthetic must be a mapping")

    batch_size = max(1, int(synth_cfg.get("batch_size", 8)))
    n_rows = max(1, int(synth_cfg.get("n_rows", 256)))
    d_in = max(1, int(synth_cfg.get("d_in", 256)))
    d_out = max(1, int(synth_cfg.get("d_out", 32)))
    x_std = float(synth_cfg.get("x_std", 1.0))
    w_std = float(synth_cfg.get("w_std", 1.0))
    kind = str(synth_cfg.get("kind", "iid_gaussian")).strip().lower()

    if x_std <= 0.0:
        raise ValueError(f"train.synthetic.x_std must be > 0, got {x_std}")
    if w_std <= 0.0:
        raise ValueError(f"train.synthetic.w_std must be > 0, got {w_std}")

    if kind == "iid_gaussian":
        x = torch.randn((batch_size, n_rows, d_in), device=device, dtype=torch.float32) * x_std
        W = torch.randn((batch_size, d_in, d_out), device=device, dtype=torch.float32) * w_std
        return x, W, W

    if kind == "masked_ab_from_x":
        return _build_masked_ab_from_x_batch(
            synth_cfg=synth_cfg,
            batch_size=batch_size,
            n_rows=n_rows,
            d_in=d_in,
            d_out=d_out,
            x_std=x_std,
            w_std=w_std,
            device=device,
        )

    if kind == "patch_stats_teacher":
        teacher_cfg = synth_cfg.get("patch_stats_teacher", {})
        if not isinstance(teacher_cfg, dict):
            raise TypeError("train.synthetic.patch_stats_teacher must be a mapping")
        signal_scale = float(teacher_cfg.get("signal_scale", w_std))
        weight_noise_std = float(teacher_cfg.get("weight_noise_std", 0.01 * w_std))
        if weight_noise_std < 0.0:
            raise ValueError(
                f"train.synthetic.patch_stats_teacher.weight_noise_std must be >= 0, got {weight_noise_std}"
            )

        x, residual_signal = _build_patch_stats_teacher_residual(
            cfg=cfg,
            teacher_cfg=teacher_cfg,
            batch_size=batch_size,
            n_rows=n_rows,
            d_in=d_in,
            d_out=d_out,
            x_std=x_std,
            device=device,
        )
        W = signal_scale * residual_signal
        if weight_noise_std > 0.0:
            W = W + weight_noise_std * torch.randn_like(W)
        return x, W, W

    if kind == "x_residual_teacher":
        teacher_cfg = synth_cfg.get("x_residual_teacher", {})
        if not isinstance(teacher_cfg, dict):
            raise TypeError("train.synthetic.x_residual_teacher must be a mapping")
        observed_w_std = float(teacher_cfg.get("observed_w_std", 0.35 * w_std))
        observed_noise_std = float(teacher_cfg.get("observed_noise_std", 0.02 * w_std))
        residual_scale = float(teacher_cfg.get("residual_scale", 1.0 * w_std))
        target_noise_std = float(teacher_cfg.get("target_noise_std", 0.0))
        if observed_w_std < 0.0:
            raise ValueError(f"train.synthetic.x_residual_teacher.observed_w_std must be >= 0, got {observed_w_std}")
        if observed_noise_std < 0.0:
            raise ValueError(
                f"train.synthetic.x_residual_teacher.observed_noise_std must be >= 0, got {observed_noise_std}"
            )
        if target_noise_std < 0.0:
            raise ValueError(
                f"train.synthetic.x_residual_teacher.target_noise_std must be >= 0, got {target_noise_std}"
            )

        x, residual_signal = _build_patch_stats_teacher_residual(
            cfg=cfg,
            teacher_cfg=teacher_cfg,
            batch_size=batch_size,
            n_rows=n_rows,
            d_in=d_in,
            d_out=d_out,
            x_std=x_std,
            device=device,
        )
        w_base = observed_w_std * torch.randn((batch_size, d_in, d_out), device=device, dtype=torch.float32)
        W_in = w_base
        if observed_noise_std > 0.0:
            W_in = W_in + observed_noise_std * torch.randn_like(W_in)
        W_target = w_base + residual_scale * residual_signal
        if target_noise_std > 0.0:
            W_target = W_target + target_noise_std * torch.randn_like(W_target)
        return x, W_in, W_target

    raise ValueError(
        "train.synthetic.kind must be one of "
        "'iid_gaussian', 'masked_ab_from_x', 'patch_stats_teacher', 'x_residual_teacher', "
        f"got {kind!r}"
    )


def _reduce_metrics(metrics: dict[str, float], device: torch.device, is_distributed: bool) -> dict[str, float]:
    if not is_distributed:
        return metrics
    keys = list(metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(values[idx].item()) for idx, key in enumerate(keys)}


def _get_encoder_conditioning_alpha_values(model: torch.nn.Module) -> list[float]:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    adapters = getattr(target, "encoder_conditioning_adapters", None)
    if adapters is None:
        return []

    alpha_values: list[float] = []
    for adapter in adapters:
        alpha = getattr(adapter, "alpha", None)
        if alpha is None:
            continue
        alpha_values.append(float(alpha.detach().reshape(-1)[0].item()))
    return alpha_values


def _get_patch_tokenizer_block_alpha_stats(model: torch.nn.Module) -> list[dict[str, float]]:
    target = model.module if isinstance(model, DDP) else model
    compiled_target = getattr(target, "_orig_mod", None)
    if compiled_target is not None:
        target = compiled_target

    patch_tokenizer = getattr(target, "patch_tokenizer", None)
    blocks = getattr(patch_tokenizer, "blocks", None)
    if blocks is None:
        return []

    out: list[dict[str, float]] = []
    for block in blocks:
        alpha = getattr(block, "alpha", None)
        if alpha is None:
            continue
        alpha_flat = alpha.detach().reshape(-1).to(dtype=torch.float32)
        if alpha_flat.numel() == 0:
            continue
        out.append(
            {
                "mean": float(alpha_flat.mean().item()),
                "abs_mean": float(alpha_flat.abs().mean().item()),
                "max_abs": float(alpha_flat.abs().max().item()),
            }
        )
    return out


def _run_worker(
    rank: int,
    world_size: int,
    cfg: dict[str, Any],
    master_addr: str,
    master_port: int,
) -> None:
    _setup_logging(cfg, rank=rank)
    logger = _logger("procedural_simple_vae_train", rank=rank)

    device = _resolve_device(cfg, rank=rank, world_size=world_size)
    backend = _resolve_backend(cfg, device=device)
    is_distributed = world_size > 1

    if is_distributed:
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    seed = int(cfg.get("data", {}).get("seed", 42)) + rank
    _seed_everything(seed)
    _set_speed_optimizations(cfg, device=device)

    model_cfg = _build_model_cfg(cfg)
    model = build_procedural_simple_vae(model_cfg).to(device)
    model = _maybe_compile(model, cfg=cfg, logger=logger)
    if is_distributed:
        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        else:
            model = DDP(
                model,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        raise TypeError("train must be a mapping")

    resume_checkpoint = str(train_cfg.get("resume_checkpoint", "")).strip()
    if resume_checkpoint:
        _load_model_weights_from_checkpoint(model, resume_checkpoint, logger)

    optimizer = _build_optimizer(model=model, cfg=cfg, device=device)
    scheduler = _build_scheduler(optimizer=optimizer, cfg=cfg)
    amp_enabled, amp_dtype = _resolve_amp(cfg=cfg, device=device)
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    max_steps = max(1, int(train_cfg.get("max_steps", 1000)))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    log_every = max(1, int(train_cfg.get("log_every", 10)))
    checkpoint_every = max(1, int(train_cfg.get("checkpoint_every", 200)))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    kl_beta = float(train_cfg.get("kl_beta", 1e-3))
    behavioral_coef = float(train_cfg.get("behavioral_coef", 1.0))
    structural_coef = float(train_cfg.get("structural_coef", 0.5))
    struct_loss_cfg = train_cfg.get("struct_loss", {})
    if not isinstance(struct_loss_cfg, dict):
        raise TypeError("train.struct_loss must be a mapping")
    struct_gamma = float(struct_loss_cfg.get("gamma", 0.5))
    struct_lambda_dir = float(struct_loss_cfg.get("lambda_dir", 1.0))
    struct_lambda_scale = float(struct_loss_cfg.get("lambda_scale", 0.25))
    struct_lambda_rec = float(struct_loss_cfg.get("lambda_rec", 0.5))
    struct_lambda_rel = float(struct_loss_cfg.get("lambda_rel", 0.1))
    struct_huber_delta = float(struct_loss_cfg.get("huber_delta", 0.1))
    use_latent_sampling = bool(model_cfg.big_vae.use_latent_sampling)

    if rank == 0:
        synth_cfg = train_cfg.get("synthetic", {})
        if not isinstance(synth_cfg, dict):
            raise TypeError("train.synthetic must be a mapping")
        logger.info("Starting procedural SimpleVAE training")
        logger.info("Resolved config:\n%s", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        logger.info(
            "Synthetic source: kind=%s batch_size=%s n_rows=%s d_in=%s d_out=%s x_std=%s w_std=%s",
            str(synth_cfg.get("kind", "iid_gaussian")),
            int(synth_cfg.get("batch_size", 8)),
            int(synth_cfg.get("n_rows", 256)),
            int(synth_cfg.get("d_in", 256)),
            int(synth_cfg.get("d_out", 32)),
            float(synth_cfg.get("x_std", 1.0)),
            float(synth_cfg.get("w_std", 1.0)),
        )
        synth_kind = str(synth_cfg.get("kind", "iid_gaussian")).strip().lower()
        if synth_kind == "masked_ab_from_x":
            teacher_cfg = synth_cfg.get("masked_ab_from_x", {})
            if not isinstance(teacher_cfg, dict):
                raise TypeError("train.synthetic.masked_ab_from_x must be a mapping")
            logger.info(
                "Synthetic masked-half task: a_std=%s c_std=%s x_noise_std=%s mask_value=%s",
                float(teacher_cfg.get("a_std", synth_cfg.get("w_std", 1.0))),
                float(teacher_cfg.get("c_std", synth_cfg.get("w_std", 1.0))),
                float(teacher_cfg.get("x_noise_std", synth_cfg.get("x_std", 1.0))),
                float(teacher_cfg.get("mask_value", 0.0)),
            )
        if synth_kind in {"patch_stats_teacher", "x_residual_teacher"}:
            teacher_key = "patch_stats_teacher" if synth_kind == "patch_stats_teacher" else "x_residual_teacher"
            teacher_cfg = synth_cfg.get(teacher_key, {})
            if not isinstance(teacher_cfg, dict):
                raise TypeError(f"train.synthetic.{teacher_key} must be a mapping")
            logger.info(
                "Synthetic teacher: patch_mean_std=%s patch_scale_min=%s patch_scale_max=%s "
                "mixture_shift_std=%s q_low=%s q_high=%s",
                float(teacher_cfg.get("patch_mean_std", synth_cfg.get("x_std", 1.0))),
                float(teacher_cfg.get("patch_scale_min", 0.5)),
                float(teacher_cfg.get("patch_scale_max", 1.5)),
                float(teacher_cfg.get("mixture_shift_std", 0.75 * float(synth_cfg.get("x_std", 1.0)))),
                float(teacher_cfg.get("q_low", 0.15)),
                float(teacher_cfg.get("q_high", 0.85)),
            )
            if synth_kind == "patch_stats_teacher":
                logger.info(
                    "Synthetic teacher target: signal_scale=%s weight_noise_std=%s",
                    float(teacher_cfg.get("signal_scale", synth_cfg.get("w_std", 1.0))),
                    float(teacher_cfg.get("weight_noise_std", 0.01 * float(synth_cfg.get("w_std", 1.0)))),
                )
            if synth_kind == "x_residual_teacher":
                logger.info(
                    "Synthetic residual task: observed_w_std=%s observed_noise_std=%s residual_scale=%s target_noise_std=%s",
                    float(teacher_cfg.get("observed_w_std", 0.35 * float(synth_cfg.get("w_std", 1.0)))),
                    float(teacher_cfg.get("observed_noise_std", 0.02 * float(synth_cfg.get("w_std", 1.0)))),
                    float(teacher_cfg.get("residual_scale", 1.0 * float(synth_cfg.get("w_std", 1.0)))),
                    float(teacher_cfg.get("target_noise_std", 0.0)),
                )
        logger.info(
            "Runtime: device=%s distributed=%s world_size=%s amp=%s compile=%s checkpoint_dir=%s",
            device,
            is_distributed,
            world_size,
            amp_enabled,
            bool(train_cfg.get("compile", False)),
            train_cfg.get("checkpoint_dir"),
        )

    t0 = time.time()
    window_start = time.time()
    loss_window = 0.0
    behavioral_window = 0.0
    structural_window = 0.0
    kl_window = 0.0
    window_steps = 0

    try:
        for step_idx in range(max_steps):
            global_step = step_idx + 1
            model.train()
            optimizer.zero_grad(set_to_none=True)

            loss_acc = 0.0
            behavioral_acc = 0.0
            structural_acc = 0.0
            kl_acc = 0.0

            for micro_idx in range(grad_accum_steps):
                sync_grad = micro_idx == grad_accum_steps - 1
                x_batch, W_input_batch, W_target_batch = _sample_procedural_batch(cfg=cfg, device=device)
                no_sync_ctx = contextlib.nullcontext()
                if is_distributed and not sync_grad:
                    no_sync_ctx = model.no_sync()  # type: ignore[union-attr]

                with no_sync_ctx:
                    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                        W_hat, mu, logvar, pred_dirs = model(W_input_batch, x_batch)
                        behavioral_loss = BigWeightVAE.operator_recon_loss(x_batch, W_target_batch, W_hat)
                        structural_loss, _ = BigWeightVAE.patch_structure_loss(
                            W_target_batch,
                            W_hat,
                            patch_size=int(model_cfg.patch_size),
                            gamma=struct_gamma,
                            lambda_dir=struct_lambda_dir,
                            lambda_scale=struct_lambda_scale,
                            lambda_rec=struct_lambda_rec,
                            lambda_rel=struct_lambda_rel,
                            huber_delta=struct_huber_delta,
                            pred_dirs=pred_dirs,
                        )
                        if use_latent_sampling:
                            kl_loss = BigWeightVAE.kl_loss(mu, logvar)
                        else:
                            kl_loss = mu.new_zeros(())

                        total_loss = mu.new_zeros(())
                        if behavioral_coef != 0.0:
                            total_loss = total_loss + behavioral_coef * behavioral_loss
                        if structural_coef != 0.0:
                            total_loss = total_loss + structural_coef * structural_loss
                        if kl_beta != 0.0:
                            total_loss = total_loss + kl_beta * kl_loss
                        loss_for_backward = total_loss / grad_accum_steps

                    if not bool(torch.isfinite(loss_for_backward.detach()).item()):
                        raise RuntimeError(
                            f"Non-finite loss at step={global_step} micro_step={micro_idx}: "
                            f"total={float(total_loss.detach().item())}"
                        )

                    if scaler.is_enabled():
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

                loss_acc += float(total_loss.detach().item())
                behavioral_acc += float(behavioral_loss.detach().item())
                structural_acc += float(structural_loss.detach().item())
                kl_acc += float(kl_loss.detach().item())

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            reduced_metrics = _reduce_metrics(
                {
                    "loss": loss_acc,
                    "behavioral": behavioral_acc,
                    "structural": structural_acc,
                    "kl": kl_acc,
                },
                device=device,
                is_distributed=is_distributed,
            )

            loss_window += reduced_metrics["loss"]
            behavioral_window += reduced_metrics["behavioral"]
            structural_window += reduced_metrics["structural"]
            kl_window += reduced_metrics["kl"]
            window_steps += 1

            if rank == 0 and global_step % log_every == 0:
                now = time.time()
                elapsed = max(now - window_start, 1e-6)
                alpha_values = _get_encoder_conditioning_alpha_values(model)
                alpha_suffix = ""
                if alpha_values:
                    alpha_suffix = " enc_alpha=[" + ",".join(f"{value:.6f}" for value in alpha_values) + "]"
                patch_alpha_stats = _get_patch_tokenizer_block_alpha_stats(model)
                logger.info(
                    "step=%s/%s loss=%.6f behavioral=%.6f structural=%.6f kl=%.6f lr=%.3e steps_per_s=%.2f elapsed=%.1fs%s",
                    global_step,
                    max_steps,
                    loss_window / window_steps,
                    behavioral_window / window_steps,
                    structural_window / window_steps,
                    kl_window / window_steps,
                    float(optimizer.param_groups[0]["lr"]),
                    window_steps / elapsed,
                    now - t0,
                    alpha_suffix,
                )
                if patch_alpha_stats:
                    patch_alpha_summary = "; ".join(
                        "block{idx}:mean={mean:.6f},abs_mean={abs_mean:.6f},max_abs={max_abs:.6f}".format(
                            idx=block_idx,
                            mean=block_stats["mean"],
                            abs_mean=block_stats["abs_mean"],
                            max_abs=block_stats["max_abs"],
                        )
                        for block_idx, block_stats in enumerate(patch_alpha_stats)
                    )
                    logger.info(
                        "patch_tok_alpha step=%s %s",
                        global_step,
                        patch_alpha_summary,
                    )
                loss_window = 0.0
                behavioral_window = 0.0
                structural_window = 0.0
                kl_window = 0.0
                window_steps = 0
                window_start = now

            if rank == 0 and (global_step % checkpoint_every == 0 or global_step == max_steps):
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg=cfg,
                    step_idx=global_step,
                    logger=logger,
                )
    finally:
        if is_distributed:
            with contextlib.suppress(Exception):
                dist.barrier()
            dist.destroy_process_group()


def _spawn_entry(
    rank: int,
    world_size: int,
    cfg: dict[str, Any],
    master_addr: str,
    master_port: int,
) -> None:
    _run_worker(
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        master_addr=master_addr,
        master_port=master_port,
    )


def _parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "conf" / "config.yaml"
    parser = argparse.ArgumentParser(description="Procedural SimpleVAE training entrypoint")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Simple dotted CLI overrides in key=value form",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(str(args.config)).expanduser().resolve()
    cfg = _load_config(config_path=config_path, overrides=list(args.overrides))
    run_artifacts = _configure_run_artifacts(cfg)

    print(
        f"[procedural_simple_vae_train] run_id={run_artifacts.get('run_id')} "
        f"artifacts_root={run_artifacts.get('root_dir')}",
        flush=True,
    )

    world_size = _resolve_world_size(cfg)
    print(f"[procedural_simple_vae_train] resolved world_size={world_size}", flush=True)

    if world_size <= 1:
        _run_worker(
            rank=0,
            world_size=1,
            cfg=cfg,
            master_addr="127.0.0.1",
            master_port=0,
        )
        return

    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        raise TypeError("train must be a mapping")

    master_addr = str(train_cfg.get("master_addr", "127.0.0.1"))
    requested_port = int(train_cfg.get("master_port", 0))
    master_port = requested_port if requested_port > 0 else _find_free_port()
    mp.spawn(
        _spawn_entry,
        args=(world_size, cfg, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
