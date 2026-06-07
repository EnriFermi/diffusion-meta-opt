from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from big_vae.eval.vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    FunctionalViTTiny,
    ViTTinyConfig,
    load_frozen_big_vae_decoder,
    make_initial_tensors,
)
from big_vae.models import BigWeightVAE
from post_train_research.big_vae_heldout_eval.evaluate_parts.decoder_adapter import (
    latent_flattening_payload_big_vae_checkpoint,
    load_latent_flattening_flow_from_checkpoint,
    paths_match,
)
from post_train_research.vit_latent_scaling.config import RunConfig
from training.big_vae_latent_diffusion import (
    load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint,
    load_frozen_layer_latent_diffusion_prior,
)


def resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def export_named_tensors(model: torch.nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    target = getattr(model, "_orig_mod", model)
    return {name: target.w(name).detach().cpu().clone() for name in names}


def load_big_vae_components(
    cfg: RunConfig,
    *,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[BigWeightVAE | None, Any | None, torch.nn.Module | None]:
    if cfg.setup.kind != "latent":
        return None, None, None
    logger.info("Loading frozen BigVAE decoder: %s", cfg.setup.big_vae_checkpoint)
    big_vae = load_frozen_big_vae_decoder(cfg.setup.big_vae_checkpoint, device=device)
    decoder_flow = None
    adapter_kind = str(cfg.setup.big_vae_decoder_adapter).strip().lower()
    if adapter_kind not in {"", "identity", "none", "off", "false"}:
        if adapter_kind not in {"latent_flattening_flow", "flow", "ir_smoothing", "latent_smoothing"}:
            raise ValueError(f"Unsupported BigVAE decoder adapter: {cfg.setup.big_vae_decoder_adapter!r}")
        logger.info("Loading BigVAE decoder adapter flow: %s", cfg.setup.big_vae_decoder_adapter_checkpoint)
        decoder_flow, flow_cfg, payload = load_latent_flattening_flow_from_checkpoint(
            checkpoint_path=cfg.setup.big_vae_decoder_adapter_checkpoint,
            model=big_vae,
            device=device,
        )
        adapter_big_vae_checkpoint = latent_flattening_payload_big_vae_checkpoint(payload)
        checkpoint_matches = bool(adapter_big_vae_checkpoint) and paths_match(
            adapter_big_vae_checkpoint,
            cfg.setup.big_vae_checkpoint,
        )
        if adapter_big_vae_checkpoint and not checkpoint_matches:
            message = (
                "BigVAE decoder adapter was trained for a different checkpoint: "
                f"adapter_big_vae_checkpoint={adapter_big_vae_checkpoint} "
                f"scaling_big_vae_checkpoint={cfg.setup.big_vae_checkpoint}"
            )
            if bool(cfg.setup.big_vae_decoder_adapter_require_checkpoint_match):
                raise ValueError(message)
            logger.warning(message)
        logger.info(
            "BigVAE decoder adapter ready: kind=%s flow_layers=%s hidden=%s depth=%s checkpoint_matches=%s",
            adapter_kind,
            int(flow_cfg.num_layers),
            int(flow_cfg.hidden_dim),
            int(flow_cfg.network_depth),
            bool(checkpoint_matches),
        )
    prior = None
    if cfg.init.kind == "diffusion_prior":
        logger.info("Loading frozen latent diffusion prior: %s", cfg.init.diffusion_prior_checkpoint)
        prior = load_frozen_layer_latent_diffusion_prior(cfg.init.diffusion_prior_checkpoint, device=device)
        loaded = load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint(
            cfg.init.diffusion_prior_checkpoint,
            big_vae=big_vae,
        )
        if loaded:
            logger.info("Loaded finetuned distribution encoder state from diffusion prior checkpoint")
    return big_vae, prior, decoder_flow


def collect_calibration_images(train_loader: DataLoader, *, num_batches: int) -> torch.Tensor | None:
    if int(num_batches) <= 0:
        return None
    images: list[torch.Tensor] = []
    for batch_idx, (batch_images, _labels) in enumerate(train_loader):
        images.append(batch_images.detach().cpu())
        if batch_idx + 1 >= int(num_batches):
            break
    if not images:
        return None
    return torch.cat(images, dim=0).contiguous()


def build_model(
    cfg: RunConfig,
    vit_cfg: ViTTinyConfig,
    initial_tensors: dict[str, torch.Tensor],
    *,
    big_vae: BigWeightVAE | None,
    prior: Any | None,
    big_vae_decoder_flow: torch.nn.Module | None = None,
) -> torch.nn.Module:
    if cfg.setup.kind == "raw":
        return FunctionalViTTiny(vit_cfg, initial_tensors, parameter_mode="direct")
    latent_init = "encoded"
    if cfg.init.kind == "fresh":
        latent_init = str(cfg.init.fresh_latent_mode)
    elif cfg.init.kind == "diffusion_prior":
        latent_init = "diffusion_prior"
    return FunctionalViTTiny(
        vit_cfg,
        initial_tensors,
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_decoder_flow=big_vae_decoder_flow,
        big_vae_latent_init=latent_init,
        big_vae_diffusion_prior=prior,
        big_vae_diffusion_prior_steps=int(cfg.init.diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.init.diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.init.diffusion_prior_eta),
        big_vae_random_init_std=float(cfg.init.random_init_std),
        big_vae_latent_noise_std=float(cfg.setup.big_vae_latent_noise_std),
        big_vae_latent_parameterization=str(cfg.setup.big_vae_latent_parameterization),
        big_vae_decode=str(cfg.setup.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.setup.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.setup.big_vae_tile_d_out),
        big_vae_encoder_context_rows=64,
        big_vae_encoder_context_std=1.0,
        big_vae_encoder_batch_size=16,
    )


def initialize_diffusion_prior(
    model: torch.nn.Module,
    calibration_images: torch.Tensor | None,
) -> None:
    if calibration_images is None:
        raise RuntimeError("calibration_images are required for diffusion prior initialization")
    target = getattr(model, "_orig_mod", model)
    if not hasattr(target, "initialize_bigvae_diffusion_prior"):
        raise TypeError("Model does not support diffusion-prior initialization")
    target.initialize_bigvae_diffusion_prior(calibration_images)


def resolve_source_checkpoint_path(
    storage_root: Path,
    *,
    source_run_dir: str,
    checkpoint_name: str,
) -> Path:
    direct = Path(source_run_dir).expanduser()
    if direct.is_file():
        return direct.resolve()
    runs_root = (storage_root / "runs").expanduser().resolve()
    if direct.is_dir():
        run_dir = direct.resolve()
    else:
        flat_candidate = (runs_root / source_run_dir).resolve()
        nested_matches = sorted(path.resolve() for path in runs_root.glob(f"*/{source_run_dir}") if path.is_dir())
        if flat_candidate.is_dir():
            run_dir = flat_candidate
        elif nested_matches:
            if len(nested_matches) > 1:
                raise FileNotFoundError(
                    f"Ambiguous source run id {source_run_dir!r}; matches: {[str(path) for path in nested_matches]}"
                )
            run_dir = nested_matches[0]
        else:
            raise FileNotFoundError(f"Could not resolve source run directory {source_run_dir!r} under {runs_root}")
    if not (run_dir / "checkpoints").is_dir():
        child_runs = sorted(path for path in run_dir.iterdir() if path.is_dir() and (path / "checkpoints").is_dir())
        if child_runs:
            run_dir = child_runs[-1].resolve()
    checkpoints_dir = run_dir / "checkpoints"
    name = str(checkpoint_name).strip()
    candidates = []
    if name.endswith(".pt"):
        candidates.append(checkpoints_dir / name)
    else:
        candidates.extend(
            [
                checkpoints_dir / f"{name}.pt",
                checkpoints_dir / f"step_{name}.pt",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve source checkpoint {checkpoint_name!r} under {run_dir}")


def load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict: {checkpoint_path}")
    payload["_checkpoint_path"] = str(checkpoint_path)
    return payload


def _copy_overlap_any(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = target.clone()
    slices = tuple(slice(0, min(int(s), int(t))) for s, t in zip(source.shape, target.shape))
    out[slices] = source[slices].to(dtype=target.dtype)
    return out


def _resize_patch_embed_weight(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out_target, in_target, h_target, w_target = target.shape
    out_source, in_source, h_source, w_source = source.shape
    copy_out = min(out_source, out_target)
    copy_in = min(in_source, in_target)
    resized = source[:copy_out, :copy_in].reshape(copy_out * copy_in, 1, h_source, w_source)
    resized = F.interpolate(resized, size=(h_target, w_target), mode="bilinear", align_corners=False)
    resized = resized.reshape(copy_out, copy_in, h_target, w_target)
    result = target.clone()
    result[:copy_out, :copy_in] = resized.to(dtype=target.dtype)
    return result


def _resize_pos_embed(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != 1 or target.shape[0] != 1:
        return _copy_overlap_any(source, target)
    src_dim = int(source.shape[-1])
    tgt_dim = int(target.shape[-1])
    src_tokens = int(source.shape[1]) - 1
    tgt_tokens = int(target.shape[1]) - 1
    src_side = int(round(src_tokens ** 0.5))
    tgt_side = int(round(tgt_tokens ** 0.5))
    if src_side * src_side != src_tokens or tgt_side * tgt_side != tgt_tokens:
        return _copy_overlap_any(source, target)
    result = target.clone()
    dim = min(src_dim, tgt_dim)
    result[:, :1, :dim] = source[:, :1, :dim].to(dtype=target.dtype)
    src_grid = source[:, 1:, :dim].reshape(1, src_side, src_side, dim).permute(0, 3, 1, 2)
    src_grid = F.interpolate(src_grid, size=(tgt_side, tgt_side), mode="bicubic", align_corners=False)
    src_grid = src_grid.permute(0, 2, 3, 1).reshape(1, tgt_tokens, dim)
    result[:, 1:, :dim] = src_grid.to(dtype=target.dtype)
    return result


def adapt_named_tensors(
    source_named_tensors: dict[str, torch.Tensor],
    target_template: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    adapted: dict[str, torch.Tensor] = {}
    exact = 0
    reshaped = 0
    missing = 0
    for name, target in target_template.items():
        source = source_named_tensors.get(name)
        if source is None:
            adapted[name] = target.detach().clone()
            missing += 1
            continue
        source = source.detach().cpu().to(dtype=target.dtype)
        if tuple(source.shape) == tuple(target.shape):
            adapted[name] = source.contiguous()
            exact += 1
            continue
        if name == "patch_embed.weight" and source.ndim == 4 and target.ndim == 4:
            adapted[name] = _resize_patch_embed_weight(source, target)
        elif name == "pos_embed":
            adapted[name] = _resize_pos_embed(source, target)
        elif source.ndim == target.ndim:
            adapted[name] = _copy_overlap_any(source, target)
        else:
            adapted[name] = target.detach().clone()
            missing += 1
            continue
        reshaped += 1
    return adapted, {"exact": exact, "reshaped": reshaped, "missing": missing}


def prepare_initial_tensors(
    cfg: RunConfig,
    *,
    storage_root: Path,
    logger: logging.Logger,
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    initial_tensors = make_initial_tensors(cfg.vit_cfg, seed=int(cfg.train.seed))
    if cfg.init.kind != "source":
        return initial_tensors, None
    checkpoint_path = resolve_source_checkpoint_path(
        storage_root,
        source_run_dir=cfg.init.source_run_dir,
        checkpoint_name=cfg.init.source_checkpoint,
    )
    payload = load_checkpoint_payload(checkpoint_path)
    source_named = payload.get("named_tensors")
    if not isinstance(source_named, dict):
        raise KeyError(f"Source checkpoint does not contain named_tensors: {checkpoint_path}")
    adapted, report = adapt_named_tensors(source_named, initial_tensors)
    logger.info(
        "Initialized from source run checkpoint %s (exact=%s reshaped=%s missing=%s)",
        checkpoint_path,
        report["exact"],
        report["reshaped"],
        report["missing"],
    )
    payload["_adapt_report"] = report
    return adapted, payload


def maybe_restore_direct_latent_state(
    model: torch.nn.Module,
    source_payload: dict[str, Any] | None,
    *,
    enabled: bool,
    logger: logging.Logger,
) -> bool:
    if not enabled or source_payload is None:
        return False
    latent_slots = source_payload.get("latent_slots")
    if not isinstance(latent_slots, dict):
        return False
    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store", None)
    if not isinstance(store, BigVAELatentTensorStore):
        return False
    try:
        store.load_materialized_latent_slots_state_dict(latent_slots, strict=True, update_radii=True)
        logger.info("Restored exact latent state directly from source checkpoint")
        return True
    except Exception as exc:
        logger.info("Direct latent restore skipped; falling back to source-weight encode (%s)", exc)
        return False
