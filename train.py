from __future__ import annotations

import contextlib
import inspect
import logging
import math
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import data_pipeline, setup_logging
from dataset.shared.collector_service import CollectorService
from dataset.shared.shared_dataset import SharedModelDataset
from dataset.shared.types import SharedSample
from models.weight_quantile_vae import EncoderConfig, ModelConfig, ResamplerConfig, WeightQuantileVAE


def _logger(name: str, rank: int) -> logging.Logger:
    return logging.getLogger(f"{name}.rank{rank}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_world_size(cfg: DictConfig) -> int:
    train_cfg = cfg.train
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


def _resolve_backend(cfg: DictConfig, device: torch.device) -> str:
    raw = str(cfg.train.get("backend", "auto")).lower()
    if raw != "auto":
        return raw
    if device.type == "cuda":
        return "nccl"
    return "gloo"


def _resolve_device(cfg: DictConfig, rank: int, world_size: int) -> torch.device:
    if torch.cuda.is_available():
        if world_size > 1:
            dev = torch.device(f"cuda:{rank}")
        else:
            wanted = str(cfg.train.get("device", "cuda:0"))
            if wanted.startswith("cuda"):
                dev = torch.device(wanted)
            else:
                dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
        return dev

    return torch.device("cpu")


def _set_speed_optimizations(cfg: DictConfig, device: torch.device) -> None:
    tf32 = bool(cfg.train.get("tf32", True))
    cudnn_benchmark = bool(cfg.train.get("cudnn_benchmark", True))

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


def _build_model_cfg(cfg: DictConfig) -> ModelConfig:
    model_cfg = cfg.get("model")
    if model_cfg is None:
        # Backward compatibility with older config layout.
        model_cfg = cfg.train.get("model", {})

    encoder_cfg = model_cfg.get("encoder", {})
    resampler_cfg = model_cfg.get("resampler", {})

    return ModelConfig(
        k=int(model_cfg.get("k", 8)),
        k_mlp=int(model_cfg.get("k_mlp", 24)),
        patch_size=int(model_cfg.get("patch_size", 16)),
        d_tok=int(model_cfg.get("d_tok", 64)),
        m_lat=int(model_cfg.get("m_lat", 32)),
        d_lat=int(model_cfg.get("d_lat", 64)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        pos_fourier_dim=int(model_cfg.get("pos_fourier_dim", 32)),
        row_mlp_mult=float(model_cfg.get("row_mlp_mult", 4.0)),
        resampler_mlp_mult=float(model_cfg.get("resampler_mlp_mult", 4.0)),
        decoder_mlp_mult=float(model_cfg.get("decoder_mlp_mult", 2.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        encoder=EncoderConfig(
            n_row_layers=int(encoder_cfg.get("n_row_layers", 2)),
            self_attn_mode=str(encoder_cfg.get("self_attn_mode", "full")),
        ),
        resampler=ResamplerConfig(
            n_layers=int(resampler_cfg.get("n_layers", 2)),
        ),
    )


def _maybe_compile(model: torch.nn.Module, cfg: DictConfig, logger: logging.Logger) -> torch.nn.Module:
    if not bool(cfg.train.get("compile", False)):
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this PyTorch version; continuing without compile")
        return model

    compile_mode = str(cfg.train.get("compile_mode", "max-autotune"))
    logger.info("Compiling model with torch.compile(mode=%s)", compile_mode)
    return torch.compile(model, mode=compile_mode, dynamic=True)


def _build_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    train_cfg = cfg.train
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    betas_cfg = train_cfg.get("betas", [0.9, 0.95])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])
    eps = float(train_cfg.get("eps", 1e-8))

    kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "eps": eps,
    }

    params = inspect.signature(torch.optim.AdamW).parameters
    if "fused" in params and device.type == "cuda":
        kwargs["fused"] = True
    if "foreach" in params:
        kwargs["foreach"] = True

    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
    warmup_steps = max(0, int(cfg.train.get("warmup_steps", 100)))
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.1))

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        progress = (step_idx - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_amp(cfg: DictConfig, device: torch.device) -> tuple[bool, torch.dtype | None]:
    mode = str(cfg.train.get("amp", "auto")).lower()
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


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: DictConfig,
    step_idx: int,
    logger: logging.Logger,
) -> None:
    checkpoint_dir = Path(str(cfg.train.get("checkpoint_dir", "./checkpoints/weight_quantile_vae")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step_idx,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    step_path = checkpoint_dir / f"step_{step_idx:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    logger.info("Checkpoint saved: %s", step_path)


def _next_valid_sample(
    dataset_iter: Iterator[SharedSample],
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    attempts = 0
    while True:
        sample = next(dataset_iter)
        x = sample.x
        W = sample.weight

        valid = (
            torch.is_tensor(x)
            and torch.is_tensor(W)
            and x.ndim == 2
            and W.ndim == 2
            and x.shape[1] == W.shape[0]
            and x.shape[0] > 0
            and W.shape[0] > 0
            and W.shape[1] > 0
        )
        if not valid:
            attempts += 1
            if attempts % 100 == 0:
                logger.warning("Skipping invalid sample repeatedly; attempts=%s", attempts)
            continue

        x = x.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
        W = W.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()

        if max_x_rows > 0 and x.shape[0] > max_x_rows:
            keep = torch.randperm(x.shape[0])[:max_x_rows]
            x = x[keep]

        return x, W


def _broadcast_tensor_2d(
    tensor: torch.Tensor | None,
    device: torch.device,
    src: int = 0,
) -> torch.Tensor:
    if not dist.is_initialized():
        if tensor is None:
            raise ValueError("tensor cannot be None when distributed is not initialized")
        return tensor.to(device=device, non_blocking=True)

    rank = dist.get_rank()
    shape = torch.zeros(2, dtype=torch.long, device=device)
    if rank == src:
        if tensor is None:
            raise ValueError("Source rank must provide tensor")
        if tensor.ndim != 2:
            raise ValueError(f"Expected rank-2 tensor, got shape {tuple(tensor.shape)}")
        shape[0] = tensor.shape[0]
        shape[1] = tensor.shape[1]

    dist.broadcast(shape, src=src)

    if rank == src:
        payload = tensor.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    else:
        payload = torch.empty((int(shape[0].item()), int(shape[1].item())), device=device, dtype=torch.float32)

    dist.broadcast(payload, src=src)
    return payload


def _fetch_batch(
    rank: int,
    device: torch.device,
    dataset_iter: Iterator[SharedSample] | None,
    use_broadcast: bool,
    max_x_rows: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_broadcast:
        x_cpu: torch.Tensor | None = None
        W_cpu: torch.Tensor | None = None
        if rank == 0:
            if dataset_iter is None:
                raise RuntimeError("rank0 requires dataset iterator in broadcast mode")
            x_cpu, W_cpu = _next_valid_sample(
                dataset_iter=dataset_iter,
                max_x_rows=max_x_rows,
                logger=logger,
            )

        x = _broadcast_tensor_2d(x_cpu, device=device, src=0)
        W = _broadcast_tensor_2d(W_cpu, device=device, src=0)
        return x, W

    if dataset_iter is None:
        raise RuntimeError("dataset iterator is required for sharded mode")

    x_cpu, W_cpu = _next_valid_sample(
        dataset_iter=dataset_iter,
        max_x_rows=max_x_rows,
        logger=logger,
    )
    return (
        x_cpu.to(device=device, non_blocking=True),
        W_cpu.to(device=device, non_blocking=True),
    )


def _run_worker(rank: int, world_size: int, cfg_dict: dict[str, Any], master_addr: str, master_port: int) -> None:
    cfg = OmegaConf.create(cfg_dict)
    setup_logging(cfg, rank=rank)
    logger = _logger("train", rank=rank)
    if rank == 0:
        logger.info("Starting training entrypoint")
        logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = _resolve_device(cfg, rank=rank, world_size=world_size)
    backend = _resolve_backend(cfg, device=device)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    seed = int(cfg.data.get("seed", 42)) + rank
    _seed_everything(seed)
    _set_speed_optimizations(cfg, device=device)

    is_distributed = world_size > 1
    streaming_mode = str(cfg.streaming.get("mode", "none")).lower()
    dataset_sharding = bool(cfg.train.get("use_dataset_sharding", True)) and is_distributed and streaming_mode != "none"
    use_broadcast = is_distributed and not dataset_sharding

    if dataset_sharding:
        cfg.streaming.distributed.enabled = True
        cfg.streaming.distributed.rank_env = "RANK"
        cfg.streaming.distributed.world_size_env = "WORLD_SIZE"
        cfg.streaming.distributed.shard_by = str(cfg.streaming.distributed.get("shard_by", "chunk"))
        base_cache_dir = str(cfg.streaming.consumer.get("local_cache_dir", "./data/streaming/cache/consumer"))
        cfg.streaming.consumer.local_cache_dir = str(Path(base_cache_dir) / f"rank_{rank}")

    logger.info(
        "Runtime: device=%s distributed=%s world_size=%s streaming_mode=%s dataset_sharding=%s",
        device,
        is_distributed,
        world_size,
        streaming_mode,
        dataset_sharding,
    )

    model = None
    optimizer = None
    scheduler = None
    scaler = None

    try:
        collector: CollectorService | None = None
        dataset: SharedModelDataset | None = None
        dataset_iter: Iterator[SharedSample] | None = None

        with contextlib.ExitStack() as stack:
            if rank == 0:
                dataset, collector = stack.enter_context(
                    data_pipeline(
                        cfg,
                        logger=logger,
                        emit_run_report=True,
                        rank=rank,
                    )
                )
                dataset_iter = iter(dataset)
            elif dataset_sharding:
                # Consumer-only wrapper over shared chunk stream.
                dataset, collector = stack.enter_context(
                    data_pipeline(
                        cfg,
                        start_collector=False,
                        predownload_models=False,
                        logger=logger,
                        emit_run_report=False,
                        rank=rank,
                    )
                )
                dataset_iter = iter(dataset)

            model_cfg = _build_model_cfg(cfg)
            model = WeightQuantileVAE(model_cfg).to(device)
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

            optimizer = _build_optimizer(model=model, cfg=cfg, device=device)
            scheduler = _build_scheduler(optimizer=optimizer, cfg=cfg)

            amp_enabled, amp_dtype = _resolve_amp(cfg=cfg, device=device)
            scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

            max_steps = max(1, int(cfg.train.get("max_steps", 1000)))
            grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
            kl_beta = float(cfg.train.get("kl_beta", 1e-3))
            grad_clip_norm = float(cfg.train.get("grad_clip_norm", 1.0))
            max_x_rows = int(cfg.train.get("max_x_rows", 0))

            log_every = max(1, int(cfg.train.get("log_every", 10)))
            checkpoint_every = max(1, int(cfg.train.get("checkpoint_every", 200)))

            loss_window = 0.0
            recon_window = 0.0
            kl_window = 0.0
            window_steps = 0
            t0 = time.time()

            for step_idx in range(max_steps):
                global_step = step_idx + 1
                model.train()
                optimizer.zero_grad(set_to_none=True)

                if rank == 0 and collector is not None and dataset is not None and not collector.is_async_mode:
                    dataset.maybe_collect(step_idx)

                loss_acc = 0.0
                recon_acc = 0.0
                kl_acc = 0.0
                step_is_finite = True

                for micro_idx in range(grad_accum_steps):
                    sync_grad = micro_idx == grad_accum_steps - 1

                    x, W = _fetch_batch(
                        rank=rank,
                        device=device,
                        dataset_iter=dataset_iter,
                        use_broadcast=use_broadcast,
                        max_x_rows=max_x_rows,
                        logger=logger,
                    )

                    no_sync_ctx = contextlib.nullcontext()
                    if is_distributed and not sync_grad:
                        no_sync_ctx = model.no_sync()  # type: ignore[union-attr]

                    with no_sync_ctx:
                        with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                            W_hat, kl_loss, _ = model(x, W)
                            recon_loss = F.mse_loss(W_hat, W)
                            total_loss = recon_loss + kl_beta * kl_loss
                            loss_for_backward = total_loss / grad_accum_steps

                        finite_flag = torch.tensor(
                            1 if torch.isfinite(loss_for_backward.detach()) else 0,
                            dtype=torch.int32,
                            device=device,
                        )
                        if is_distributed:
                            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

                        if int(finite_flag.item()) == 0:
                            step_is_finite = False
                        else:
                            if scaler.is_enabled():
                                scaler.scale(loss_for_backward).backward()
                            else:
                                loss_for_backward.backward()

                    loss_acc += float(total_loss.detach().item())
                    recon_acc += float(recon_loss.detach().item())
                    kl_acc += float(kl_loss.detach().item())

                    if not step_is_finite:
                        break

                if not step_is_finite:
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        logger.warning("Skipping step %s due to non-finite loss", global_step)
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()

                step_loss = loss_acc / grad_accum_steps
                step_recon = recon_acc / grad_accum_steps
                step_kl = kl_acc / grad_accum_steps

                stats = torch.tensor([step_loss, step_recon, step_kl], dtype=torch.float32, device=device)
                if is_distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    stats /= float(world_size)

                loss_window += float(stats[0].item())
                recon_window += float(stats[1].item())
                kl_window += float(stats[2].item())
                window_steps += 1

                if rank == 0 and global_step % log_every == 0:
                    dt = max(1e-6, time.time() - t0)
                    avg_loss = loss_window / max(1, window_steps)
                    avg_recon = recon_window / max(1, window_steps)
                    avg_kl = kl_window / max(1, window_steps)
                    lr = float(optimizer.param_groups[0]["lr"])
                    speed = window_steps / dt

                    cache_metric = dataset.cache_size() if dataset is not None else 0
                    logger.info(
                        "step=%s/%s loss=%.6f recon=%.6f kl=%.6f lr=%.6e steps/s=%.2f cache=%s",
                        global_step,
                        max_steps,
                        avg_loss,
                        avg_recon,
                        avg_kl,
                        lr,
                        speed,
                        cache_metric,
                    )

                    loss_window = 0.0
                    recon_window = 0.0
                    kl_window = 0.0
                    window_steps = 0
                    t0 = time.time()

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

            if rank == 0:
                logger.info("Training completed successfully: steps=%s", max_steps)

    finally:
        if is_distributed:
            try:
                dist.barrier()
            except Exception:
                pass

        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def _spawn_entry(rank: int, world_size: int, cfg_dict: dict[str, Any], master_addr: str, master_port: int) -> None:
    _run_worker(rank=rank, world_size=world_size, cfg_dict=cfg_dict, master_addr=master_addr, master_port=master_port)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    world_size = _resolve_world_size(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    if world_size <= 1:
        _run_worker(rank=0, world_size=1, cfg_dict=cfg_dict, master_addr="127.0.0.1", master_port=0)
        return

    master_addr = str(cfg.train.get("master_addr", "127.0.0.1"))
    requested_port = int(cfg.train.get("master_port", 0))
    master_port = requested_port if requested_port > 0 else _find_free_port()

    mp.spawn(
        _spawn_entry,
        args=(world_size, cfg_dict, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
