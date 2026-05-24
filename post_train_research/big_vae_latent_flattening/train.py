from __future__ import annotations

import contextlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dataset.big_vae_offline import OfflineBigVAEDataset
from experiments.train_big_vae import (
    SourceSampleRecord,
    _build_model_cfg,
    _build_training_batch_from_source_states,
    _make_source_slice_state,
    _normalize_model_state_dict_keys,
)
from models.weight_quantile_vae import build_weight_quantile_vae
from post_train_research.big_vae_latent_flattening.config import RunConfig
from post_train_research.big_vae_latent_flattening.flow import RealNVPConfig, RealNVPFlow
from post_train_research.big_vae_latent_flattening.geometry import relaxed_distortion_measure
from post_train_research.big_vae_latent_flattening.runtime import RunPaths, append_metrics_row, resolve_path, save_checkpoint


@dataclass(slots=True)
class PreparedBatch:
    W: torch.Tensor
    X: torch.Tensor
    x_mask: torch.Tensor
    d_in_mask: torch.Tensor
    d_out_mask: torch.Tensor


@dataclass(slots=True)
class EncodedDecoderState:
    z: torch.Tensor
    logvar: torch.Tensor
    dist_patch_by_patch: torch.Tensor | None
    patch_mask: torch.Tensor
    d_in_mask: torch.Tensor
    d_out_mask: torch.Tensor
    d_in: int
    d_out: int
    d_in_pad: int
    T: int


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def configure_torch_runtime(cfg: RunConfig, device: torch.device) -> None:
    if device.type == "cuda" and bool(cfg.train.tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def maybe_autocast(device: torch.device, enabled: bool) -> Iterator[None]:
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
        return
    yield


def load_frozen_big_vae(checkpoint_path: str | Path, *, device: torch.device) -> nn.Module:
    path = resolve_path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)!r}: {path}")
    cfg_payload = payload.get("config")
    if not isinstance(cfg_payload, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'config': {path}")
    model_cfg = _build_model_cfg(OmegaConf.create(cfg_payload))
    model = build_weight_quantile_vae(model_cfg).to(device)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'model_state': {path}")
    model.load_state_dict(_normalize_model_state_dict_keys(state), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    required = ("_encode_distribution_context", "_encode_latent_slots", "_decode_from_decoder_latent")
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            "BigVAE latent flattening currently requires the full BigWeightVAE private decode API; "
            f"missing methods: {missing}"
        )
    return model


def build_offline_dataset(cfg: RunConfig, *, patch_size: int) -> OfflineBigVAEDataset:
    min_d_in = int(cfg.data.max_T_patches) * int(patch_size)
    return OfflineBigVAEDataset(
        root_dir=resolve_path(cfg.data.offline_root),
        shuffle_chunks=bool(cfg.data.shuffle_chunks),
        shuffle_records_within_chunk=bool(cfg.data.shuffle_records_within_chunk),
        repeat=bool(cfg.data.repeat),
        seed=int(cfg.data.seed),
        shard_rank=0,
        shard_world_size=1,
        shard_by_chunk=False,
        weight_cache_size=int(cfg.data.weight_cache_size),
        sampling_mode=str(cfg.data.sampling_mode),
        sampling_group_keys=list(cfg.data.sampling_group_keys),
        sampling_window_size=int(cfg.data.sampling_window_size_records),
        sampling_max_records_per_chunk_round=int(cfg.data.sampling_max_records_per_chunk_round),
        x_chunk_cache_size=int(cfg.data.x_chunk_cache_size),
        runtime_enforce_stage_compatibility=bool(cfg.data.runtime_enforce_stage_compatibility),
        runtime_min_d_in=int(min_d_in),
        runtime_min_d_out=int(cfg.data.max_d_out),
    )


def _source_record_from_sample(sample: Any, device: torch.device) -> SourceSampleRecord:
    x = getattr(sample, "x", None)
    W = getattr(sample, "weight", None)
    if not (
        torch.is_tensor(x)
        and torch.is_tensor(W)
        and x.ndim == 2
        and W.ndim == 2
        and int(x.shape[1]) == int(W.shape[0])
    ):
        raise ValueError(
            "offline sample has invalid x/W shapes: "
            f"x={tuple(x.shape) if torch.is_tensor(x) else type(x)} "
            f"W={tuple(W.shape) if torch.is_tensor(W) else type(W)}"
        )
    return SourceSampleRecord(
        x=x.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        W=W.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        model_name=str(getattr(sample, "model_name", "") or "").strip(),
        layer_name=str(getattr(sample, "layer_name", "") or "").strip(),
    )


def next_prepared_batch(
    *,
    dataset_iter: Iterator[Any],
    cfg: RunConfig,
    device: torch.device,
    patch_size: int,
) -> PreparedBatch:
    source_states = []
    needed = max(int(cfg.data.batch_size), int(cfg.data.source_pool_size))
    attempts = 0
    max_attempts = max(needed * 32, 128)
    while len(source_states) < needed:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to build a source pool for BigVAE flattening batch: "
                f"needed={needed} collected={len(source_states)} attempts={attempts}"
            )
        try:
            sample = next(dataset_iter)
        except StopIteration as exc:
            raise RuntimeError("Offline BigVAE dataset ended before max_steps; set data.repeat=true") from exc
        try:
            record = _source_record_from_sample(sample, device)
            state = _make_source_slice_state(
                record,
                max_T_patches=int(cfg.data.max_T_patches),
                max_d_out=int(cfg.data.max_d_out),
                patch_size=int(patch_size),
            )
        except (RuntimeError, ValueError):
            continue
        source_states.append(state)

    target_x_rows = int(cfg.data.max_x_rows) if int(cfg.data.max_x_rows) > 0 else None
    target_d_in = int(patch_size) * int(cfg.data.max_T_patches) if bool(cfg.data.stable_batch_shapes) else None
    target_d_out = int(cfg.data.max_d_out) if bool(cfg.data.stable_batch_shapes) else None
    consumed = _build_training_batch_from_source_states(
        source_states,
        batch_size=int(cfg.data.batch_size),
        start_offset=random.randrange(len(source_states)),
        target_x_rows=target_x_rows,
        target_d_in=target_d_in,
        target_d_out=target_d_out,
    )
    return PreparedBatch(
        W=consumed.W,
        X=consumed.x,
        x_mask=consumed.x_mask,
        d_in_mask=consumed.d_in_mask,
        d_out_mask=consumed.d_out_mask,
    )


@torch.no_grad()
def encode_decoder_state(
    *,
    big_vae: nn.Module,
    batch: PreparedBatch,
    device: torch.device,
    amp_encode: bool,
) -> EncodedDecoderState:
    W = batch.W.to(device=device, dtype=torch.float32, non_blocking=True)
    X = batch.X.to(device=device, dtype=torch.float32, non_blocking=True)
    x_mask = batch.x_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    d_in_mask = batch.d_in_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    d_out_mask = batch.d_out_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    B, d_in, d_out = W.shape
    with maybe_autocast(device, amp_encode):
        validated_d_in_mask = big_vae._validate_d_in_mask(d_in_mask, batch_size=B, d_in=d_in, device=device)
        validated_d_out_mask = big_vae._validate_d_out_mask(d_out_mask, batch_size=B, d_out=d_out, device=device)
        T, d_in_pad, patch_mask, _structural_patch_mask, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled = (
            big_vae._encode_distribution_context(
                X,
                x_mask=x_mask,
                d_in_mask=validated_d_in_mask,
            )
        )
        latent_slots = big_vae._encode_latent_slots(
            W,
            T=T,
            d_in_pad=d_in_pad,
            patch_mask=patch_mask,
            d_out_mask=validated_d_out_mask,
            dist_var_by_patch=dist_var_by_patch,
            dist_patch_by_patch=dist_patch_by_patch,
            dist_var_pooled=dist_var_pooled,
            return_debug_info=False,
        )
        base_z = big_vae.latent_norm(latent_slots.reshape(B, big_vae.flat_lat_dim))
        mu_input_z = big_vae._mu_head_input_z(latent_slots)
        decoder_z, _mu, logvar = big_vae._sample_latent_posterior(base_z, mu_input_z=mu_input_z)

    return EncodedDecoderState(
        z=decoder_z.detach().to(dtype=torch.float32),
        logvar=logvar.detach().to(dtype=torch.float32),
        dist_patch_by_patch=dist_patch_by_patch.detach().to(dtype=torch.float32) if dist_patch_by_patch is not None else None,
        patch_mask=patch_mask.detach(),
        d_in_mask=validated_d_in_mask.detach(),
        d_out_mask=validated_d_out_mask.detach(),
        d_in=int(d_in),
        d_out=int(d_out),
        d_in_pad=int(d_in_pad),
        T=int(T),
    )


def build_masked_flattened_decoder(
    *,
    big_vae: nn.Module,
    flow: RealNVPFlow,
    state: EncodedDecoderState,
) -> Callable[[torch.Tensor], torch.Tensor]:
    valid = state.d_in_mask.unsqueeze(-1).to(dtype=torch.float32) * state.d_out_mask.unsqueeze(1).to(dtype=torch.float32)

    def decode(z_flat: torch.Tensor) -> torch.Tensor:
        decoder_z, _inverse_log_det = flow.inverse(z_flat)
        W_hat = big_vae._decode_from_decoder_latent(
            decoder_z,
            dist_patch_by_patch=state.dist_patch_by_patch,
            patch_mask=state.patch_mask,
            d_in_mask=state.d_in_mask,
            d_out_mask=state.d_out_mask,
            d_in=int(state.d_in),
            d_out=int(state.d_out),
            d_in_pad=int(state.d_in_pad),
            T=int(state.T),
        )[0]
        return W_hat.to(dtype=torch.float32) * valid

    return decode


def build_optimizer(flow: nn.Module, cfg: RunConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        flow.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        eps=float(cfg.train.adam_eps),
    )


def run_training(cfg: RunConfig, paths: RunPaths, logger: logging.Logger) -> dict[str, Any]:
    seed_everything(int(cfg.train.seed))
    device = resolve_device(cfg.train.device)
    configure_torch_runtime(cfg, device)
    logger.info("Loading frozen BigVAE checkpoint: %s", cfg.big_vae.checkpoint)
    big_vae = load_frozen_big_vae(cfg.big_vae.checkpoint, device=device)
    latent_dim = int(getattr(big_vae, "flat_lat_dim"))
    patch_size = int(big_vae.cfg.patch_size)
    logger.info("BigVAE ready: latent_dim=%s patch_size=%s device=%s", latent_dim, patch_size, device)

    flow = RealNVPFlow(
        RealNVPConfig(
            dim=latent_dim,
            num_layers=int(cfg.flow.num_layers),
            hidden_dim=int(cfg.flow.hidden_dim),
            network_depth=int(cfg.flow.network_depth),
            log_scale_clamp=float(cfg.flow.log_scale_clamp),
            dropout=float(cfg.flow.dropout),
        )
    ).to(device)
    optimizer = build_optimizer(flow, cfg)
    logger.info(
        "Flow ready: layers=%s hidden=%s depth=%s trainable_params=%s",
        int(cfg.flow.num_layers),
        int(cfg.flow.hidden_dim),
        int(cfg.flow.network_depth),
        sum(param.numel() for param in flow.parameters() if param.requires_grad),
    )

    dataset = build_offline_dataset(cfg, patch_size=patch_size)
    dataset_iter = iter(dataset)
    last_metrics: dict[str, Any] = {}
    try:
        for step in range(1, int(cfg.train.max_steps) + 1):
            flow.train()
            batch = next_prepared_batch(
                dataset_iter=dataset_iter,
                cfg=cfg,
                device=device,
                patch_size=patch_size,
            )
            state = encode_decoder_state(
                big_vae=big_vae,
                batch=batch,
                device=device,
                amp_encode=bool(cfg.train.amp_encode),
            )
            z_prime, forward_log_det = flow(state.z)
            decode = build_masked_flattened_decoder(big_vae=big_vae, flow=flow, state=state)
            distortion = relaxed_distortion_measure(
                decode,
                z_prime,
                eta=float(cfg.train.eta),
                probes=int(cfg.train.probes),
                loss_scale=str(cfg.train.loss_scale),
                create_graph=True,
            )
            z_norm_loss = z_prime.pow(2).sum(dim=1).mean()
            loss = float(cfg.train.iso_coef) * distortion.loss + float(cfg.train.z_norm_coef) * z_norm_loss
            with torch.no_grad():
                z_roundtrip, _roundtrip_log_det = flow.inverse(z_prime.detach())
                flow_cycle_mse = (z_roundtrip - state.z).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = 0.0
            if float(cfg.train.grad_clip_norm) > 0.0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=float(cfg.train.grad_clip_norm)).item()
                )
            optimizer.step()

            last_metrics = {
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "iso_loss": float(distortion.loss.detach().cpu().item()),
                "iso_ratio": float(distortion.ratio.cpu().item()),
                "iso_zero_baseline": float(distortion.zero_baseline.cpu().item()),
                "trace_g": float(distortion.trace_g.cpu().item()),
                "trace_g2": float(distortion.trace_g2.cpu().item()),
                "z_norm_loss": float(z_norm_loss.detach().cpu().item()),
                "flow_cycle_mse": float(flow_cycle_mse.cpu().item()),
                "flow_logdet_mean": float(forward_log_det.detach().mean().cpu().item()),
                "grad_norm": float(grad_norm),
                "batch_size": int(state.z.shape[0]),
                "latent_dim": int(state.z.shape[1]),
                "d_in": int(state.d_in),
                "d_out": int(state.d_out),
                "T": int(state.T),
            }
            append_metrics_row(paths.metrics_csv, last_metrics)

            if step % max(1, int(cfg.train.log_every_steps)) == 0 or step == 1:
                logger.info(
                    "step=%s loss=%.6g iso=%.6g ratio=%.6g z_norm=%.6g grad=%.4g shape=(B%s,%s,%s)",
                    step,
                    last_metrics["loss"],
                    last_metrics["iso_loss"],
                    last_metrics["iso_ratio"],
                    last_metrics["z_norm_loss"],
                    last_metrics["grad_norm"],
                    last_metrics["batch_size"],
                    last_metrics["d_in"],
                    last_metrics["d_out"],
                )

            checkpoint_due = int(cfg.storage.checkpoint_every_steps) > 0 and step % int(cfg.storage.checkpoint_every_steps) == 0
            if checkpoint_due:
                save_checkpoint(
                    paths.checkpoints_dir / f"step_{step:06d}.pt",
                    cfg=cfg,
                    flow=flow,
                    optimizer=optimizer,
                    step=step,
                    metrics=last_metrics,
                    big_vae_checkpoint=cfg.big_vae.checkpoint,
                )
                if bool(cfg.storage.save_latest):
                    save_checkpoint(
                        paths.checkpoints_dir / "latest.pt",
                        cfg=cfg,
                        flow=flow,
                        optimizer=optimizer,
                        step=step,
                        metrics=last_metrics,
                        big_vae_checkpoint=cfg.big_vae.checkpoint,
                    )

        if bool(cfg.storage.save_final):
            save_checkpoint(
                paths.checkpoints_dir / "final.pt",
                cfg=cfg,
                flow=flow,
                optimizer=optimizer,
                step=int(cfg.train.max_steps),
                metrics=last_metrics,
                big_vae_checkpoint=cfg.big_vae.checkpoint,
            )
        if bool(cfg.storage.save_latest):
            save_checkpoint(
                paths.checkpoints_dir / "latest.pt",
                cfg=cfg,
                flow=flow,
                optimizer=optimizer,
                step=int(cfg.train.max_steps),
                metrics=last_metrics,
                big_vae_checkpoint=cfg.big_vae.checkpoint,
            )
    finally:
        dataset.close()

    return {
        "run_dir": str(paths.run_dir),
        "checkpoint": str(paths.checkpoints_dir / "final.pt"),
        "last_metrics": last_metrics,
    }
