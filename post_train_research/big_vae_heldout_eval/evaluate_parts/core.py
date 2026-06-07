from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from ..common import METRIC_NAMES, env_bool, tensor_to_float
from big_vae.datasets.offline import infer_layer_depth, infer_layer_type
from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from training.big_vae.checkpointing import _normalize_model_state_dict_keys
from training.big_vae.data_types import SourceSampleRecord
from training.big_vae.runtime import _autocast_context, _build_model_cfg, _resolve_amp
from training.big_vae.source_batching import _build_training_batch_from_source_states
from training.big_vae.source_pool import _make_source_slice_state
from training.runtime import resolve_device as runtime_resolve_device
from .decoder_adapter import IdentityDecoderAdapter


def _compute_model_latent_kl(model: torch.nn.Module, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    latent_kl = getattr(model, "latent_kl_loss", None)
    if callable(latent_kl):
        return latent_kl(mu, logvar)
    return WeightQuantileVAE.kl_loss(mu, logvar)


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, DictConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)!r}: {checkpoint_path}")
    cfg_payload = payload.get("config")
    if not isinstance(cfg_payload, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'config': {checkpoint_path}")
    ckpt_cfg = OmegaConf.create(cfg_payload)
    cfg = _build_model_cfg(ckpt_cfg)
    cfg.big_vae.use_latent_sampling=False
    print(cfg)
    model = build_weight_quantile_vae(cfg).to(device)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'model_state': {checkpoint_path}")
    model.load_state_dict(_normalize_model_state_dict_keys(state), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt_cfg, payload


def _resolve_eval_device(runtime_cfg: DictConfig) -> torch.device:
    env_device = str(os.environ.get("EVAL_DEVICE", "")).strip()
    if env_device:
        return torch.device(env_device)
    return runtime_resolve_device(runtime_cfg, rank=0, world_size=1, section="train")


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


def _loss_coefficients(cfg: DictConfig) -> dict[str, float]:
    train_cfg = cfg.get("train", {})
    behavioral_cfg = train_cfg.get("behavioral_loss", {})
    if behavioral_cfg is None:
        behavioral_cfg = {}
    struct_cfg = train_cfg.get("struct_loss", {})
    if struct_cfg is None:
        struct_cfg = {}
    return {
        "behavioral_coef": float(train_cfg.get("behavioral_coef", 1.0)),
        "structural_coef": float(train_cfg.get("structural_coef", 0.5)),
        "kl_beta": float(train_cfg.get("kl_beta", 0.0)),
        "behavioral_lambda_operator": float(behavioral_cfg.get("lambda_operator", 1.0)),
        "behavioral_lambda_dir": float(behavioral_cfg.get("lambda_dir", 0.0)),
        "behavioral_lambda_scale": float(behavioral_cfg.get("lambda_scale", 0.0)),
        "behavioral_gamma": float(behavioral_cfg.get("gamma", 0.5)),
        "behavioral_huber_delta": float(behavioral_cfg.get("huber_delta", 0.1)),
        "struct_lambda_dir": float(struct_cfg.get("lambda_dir", 1.0)),
        "struct_lambda_scale": float(struct_cfg.get("lambda_scale", 0.25)),
        "struct_lambda_rec": float(struct_cfg.get("lambda_rec", 0.5)),
        "struct_lambda_rel": float(struct_cfg.get("lambda_rel", 0.1)),
        "struct_gamma": float(struct_cfg.get("gamma", 0.5)),
        "struct_huber_delta": float(struct_cfg.get("huber_delta", 0.1)),
    }


@torch.no_grad()
def _compute_loss_metrics(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    W_s: torch.Tensor,
    x_s: torch.Tensor,
    x_mask_s: torch.Tensor,
    d_in_mask_s: torch.Tensor,
    d_out_mask_s: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    decoder_adapter: IdentityDecoderAdapter | None = None,
) -> dict[str, float]:
    coeffs = _loss_coefficients(cfg)
    patch_size = int(cfg.model.get("patch_size", 16))
    use_latent_sampling = bool(getattr(model.cfg.big_vae, "use_latent_sampling", False))
    adapter = decoder_adapter or IdentityDecoderAdapter()

    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        decoded = adapter.decode(
            model=model,
            W_s=W_s,
            x_s=x_s,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        W_hat = decoded.W_hat
        mu = decoded.mu
        logvar = decoded.logvar
        pred_dirs = decoded.pred_dirs
        behavioral_operator = WeightQuantileVAE.operator_recon_loss(
            x_s,
            W_s,
            W_hat,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        behavioral_dir, behavioral_scale = WeightQuantileVAE.operator_direction_scale_loss(
            x_s,
            W_s,
            W_hat,
            x_mask=x_mask_s,
            d_out_mask=d_out_mask_s,
            gamma=coeffs["behavioral_gamma"],
            huber_delta=coeffs["behavioral_huber_delta"],
        )
        _struct_all, struct_details = WeightQuantileVAE.patch_structure_loss(
            W_s,
            W_hat,
            patch_size=patch_size,
            gamma=coeffs["struct_gamma"],
            lambda_dir=1.0,
            lambda_scale=1.0,
            lambda_rec=1.0,
            lambda_rel=1.0,
            huber_delta=coeffs["struct_huber_delta"],
            pred_dirs=pred_dirs,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        struct_dir = struct_details["L_dir"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        struct_scale = struct_details["L_scale"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        struct_rec = struct_details["L_rec"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        struct_rel = struct_details["L_rel"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        behavioral_loss = (
            coeffs["behavioral_lambda_operator"] * behavioral_operator
            + coeffs["behavioral_lambda_dir"] * behavioral_dir
            + coeffs["behavioral_lambda_scale"] * behavioral_scale
        )
        structural_loss = (
            coeffs["struct_lambda_dir"] * struct_dir
            + coeffs["struct_lambda_scale"] * struct_scale
            + coeffs["struct_lambda_rec"] * struct_rec
            + coeffs["struct_lambda_rel"] * struct_rel
        )
        kl_loss = _compute_model_latent_kl(model, mu, logvar) if use_latent_sampling else behavioral_operator.new_zeros(())
        total_loss = (
            coeffs["behavioral_coef"] * behavioral_loss
            + coeffs["structural_coef"] * structural_loss
            + coeffs["kl_beta"] * kl_loss
        )

    return {
        "total_loss": tensor_to_float(total_loss),
        "behavioral_loss": tensor_to_float(behavioral_loss),
        "behavioral_operator": tensor_to_float(behavioral_operator),
        "behavioral_dir": tensor_to_float(behavioral_dir),
        "behavioral_scale": tensor_to_float(behavioral_scale),
        "structural_loss": tensor_to_float(structural_loss),
        "struct_dir": tensor_to_float(struct_dir),
        "struct_scale": tensor_to_float(struct_scale),
        "struct_rec": tensor_to_float(struct_rec),
        "struct_rel": tensor_to_float(struct_rel),
        "kl_loss": tensor_to_float(kl_loss),
        **{key: float(value) for key, value in decoded.metrics.items()},
    }


def _record_metrics_writer(path: Path) -> tuple[csv.DictWriter, Any] | tuple[None, None]:
    if not env_bool("EVAL_SAVE_RECORD_METRICS", True):
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8", newline="")
    fieldnames = [
        "record_index",
        "dataset",
        "model",
        "layer",
        "weight_shape",
        "x_shape",
        "batch_slices",
        "source_slices_total",
        "source_batch_index",
        "finite",
        *METRIC_NAMES,
    ]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return writer, fh
