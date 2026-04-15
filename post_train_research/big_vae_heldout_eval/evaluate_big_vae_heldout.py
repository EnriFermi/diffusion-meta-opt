from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from hydra import compose, initialize_config_dir
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    GroupedMetrics,
    env_bool,
    env_int,
    env_path,
    finite_metrics,
    primary_dataset_name,
    promote_run_profile_to_root,
    sanitize_programmatic_hydra_logging,
    tensor_to_float,
    write_csv,
    write_json,
)
from dataset.big_vae_offline import OfflineBigVAEDataset
from dataset.logging_utils import configure_process_logging
from experiments.train_big_vae import (
    SourceSampleRecord,
    _autocast_context,
    _build_model_cfg,
    _build_training_batch_from_source_states,
    _compute_curriculum_slice_sizes,
    _make_source_slice_state,
    _normalize_model_state_dict_keys,
    _remaining_source_state_slices,
    _resolve_amp,
    _stable_batch_shape_targets,
)
from models.weight_quantile_vae import WeightQuantileVAE, build_weight_quantile_vae
from training.runtime import resolve_device as runtime_resolve_device


LOGGER = logging.getLogger("evaluate_big_vae_heldout")


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, DictConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)!r}: {checkpoint_path}")
    cfg_payload = payload.get("config")
    if not isinstance(cfg_payload, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'config': {checkpoint_path}")
    ckpt_cfg = OmegaConf.create(cfg_payload)
    model = build_weight_quantile_vae(_build_model_cfg(ckpt_cfg)).to(device)
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
) -> dict[str, float]:
    coeffs = _loss_coefficients(cfg)
    patch_size = int(cfg.model.get("patch_size", 16))
    use_latent_sampling = bool(getattr(model.cfg.big_vae, "use_latent_sampling", False))

    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        W_hat, mu, logvar, pred_dirs = model(
            W_s,
            x_s,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
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
        kl_loss = WeightQuantileVAE.kl_loss(mu, logvar) if use_latent_sampling else behavioral_operator.new_zeros(())
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
        "total_loss",
        "behavioral_loss",
        "behavioral_operator",
        "behavioral_dir",
        "behavioral_scale",
        "structural_loss",
        "struct_dir",
        "struct_scale",
        "struct_rec",
        "struct_rel",
        "kl_loss",
    ]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return writer, fh


def _evaluate_dataset(
    *,
    model: torch.nn.Module,
    ckpt_cfg: DictConfig,
    dataset: OfflineBigVAEDataset,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    eval_batch_size = max(1, env_int("EVAL_BATCH_SIZE", int(ckpt_cfg.train.get("slice_batch_size", 1))))
    max_records = max(0, env_int("EVAL_MAX_RECORDS", 0))
    max_slices_per_source = max(0, env_int("EVAL_MAX_SLICES_PER_SOURCE", 0))
    log_every_records = max(1, env_int("EVAL_LOG_EVERY_RECORDS", 100))
    seed = env_int("EVAL_SEED", int(ckpt_cfg.data.get("seed", 42)) if "data" in ckpt_cfg else 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if env_int("EVAL_STAGE", 0) > 0:
        with open_dict(ckpt_cfg):
            ckpt_cfg.train.stage = int(env_int("EVAL_STAGE", 0))

    max_T_patches, max_d_out = _compute_curriculum_slice_sizes(ckpt_cfg)
    patch_size = int(ckpt_cfg.model.get("patch_size", 16))
    max_x_rows = max(0, int(ckpt_cfg.train.get("max_x_rows", 0)))
    target_x_rows, target_d_in, target_d_out = _stable_batch_shape_targets(
        cfg=ckpt_cfg,
        patch_size=patch_size,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
        max_x_rows=max_x_rows,
    )
    amp_enabled, amp_dtype = _resolve_amp(ckpt_cfg, device)
    amp_enabled = bool(env_bool("EVAL_AMP", amp_enabled))

    grouped = GroupedMetrics()
    skipped: dict[str, int] = {"invalid": 0, "incompatible": 0, "non_finite": 0}
    record_writer, record_fh = _record_metrics_writer(output_dir / "record_metrics.csv")
    records_seen = 0
    records_evaluated = 0
    slices_evaluated = 0

    try:
        for sample in dataset:
            if max_records > 0 and records_seen >= max_records:
                break
            records_seen += 1
            dataset_name = primary_dataset_name(getattr(sample, "meta", {}) or {})
            model_name = str(getattr(sample, "model_name", "") or "").strip() or "<unknown_model>"
            layer_name = str(getattr(sample, "layer_name", "") or "").strip() or "<unknown_layer>"
            try:
                source = _source_record_from_sample(sample, device)
                state = _make_source_slice_state(
                    source,
                    max_T_patches=max_T_patches,
                    max_d_out=max_d_out,
                    patch_size=patch_size,
                )
            except Exception:
                skipped["invalid"] += 1
                continue

            total_source_slices = _remaining_source_state_slices(state)
            if total_source_slices <= 0:
                skipped["incompatible"] += 1
                continue
            if max_slices_per_source > 0:
                total_to_eval = min(total_source_slices, max_slices_per_source)
            else:
                total_to_eval = total_source_slices

            source_batch_idx = 0
            evaluated_for_source = 0
            while evaluated_for_source < total_to_eval:
                remaining = _remaining_source_state_slices(state)
                if remaining <= 0:
                    break
                batch_slices = min(eval_batch_size, remaining, total_to_eval - evaluated_for_source)
                batch_payload = _build_training_batch_from_source_states(
                    [state],
                    batch_size=batch_slices,
                    target_x_rows=target_x_rows,
                    target_d_in=target_d_in,
                    target_d_out=target_d_out,
                )
                metrics = _compute_loss_metrics(
                    model=model,
                    cfg=ckpt_cfg,
                    W_s=batch_payload.W.to(device=device, non_blocking=True),
                    x_s=batch_payload.x.to(device=device, non_blocking=True),
                    x_mask_s=batch_payload.x_mask.to(device=device, non_blocking=True),
                    d_in_mask_s=batch_payload.d_in_mask.to(device=device, non_blocking=True),
                    d_out_mask_s=batch_payload.d_out_mask.to(device=device, non_blocking=True),
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                if not finite_metrics(metrics):
                    skipped["non_finite"] += 1
                else:
                    grouped.update(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        layer_name=layer_name,
                        metrics=metrics,
                        weight=batch_slices,
                    )
                    slices_evaluated += int(batch_slices)

                if record_writer is not None:
                    record_writer.writerow(
                        {
                            "record_index": int(records_seen),
                            "dataset": dataset_name,
                            "model": model_name,
                            "layer": layer_name,
                            "weight_shape": "x".join(str(int(dim)) for dim in source.W.shape),
                            "x_shape": "x".join(str(int(dim)) for dim in source.x.shape),
                            "batch_slices": int(batch_slices),
                            "source_slices_total": int(total_source_slices),
                            "source_batch_index": int(source_batch_idx),
                            "finite": bool(finite_metrics(metrics)),
                            **metrics,
                        }
                    )

                evaluated_for_source += int(batch_slices)
                source_batch_idx += 1

            records_evaluated += 1
            if records_evaluated % log_every_records == 0:
                LOGGER.info(
                    "Eval progress: records_seen=%s records_evaluated=%s slices_evaluated=%s skipped=%s",
                    records_seen,
                    records_evaluated,
                    slices_evaluated,
                    skipped,
                )
    finally:
        if record_fh is not None:
            record_fh.close()

    payload = grouped.payload()
    payload["run"] = {
        "records_seen": int(records_seen),
        "records_evaluated": int(records_evaluated),
        "slices_evaluated": int(slices_evaluated),
        "skipped": {key: int(value) for key, value in skipped.items()},
        "eval_batch_size": int(eval_batch_size),
        "max_records": int(max_records),
        "max_slices_per_source": int(max_slices_per_source),
        "stage": int(ckpt_cfg.train.get("stage", 1)),
        "max_T_patches": int(max_T_patches),
        "max_d_out": int(max_d_out),
        "patch_size": int(patch_size),
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": str(amp_dtype),
        "device": str(device),
    }
    return payload


def _load_base_config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        return compose(config_name="config")


def main() -> None:
    cfg = _load_base_config()
    promote_run_profile_to_root(cfg)
    sanitize_programmatic_hydra_logging(cfg, role="evaluate_big_vae_heldout")
    raw_checkpoint = str(os.environ.get("BIG_VAE_CHECKPOINT", "")).strip()
    if not raw_checkpoint:
        raise ValueError("Set BIG_VAE_CHECKPOINT=/path/to/big_vae_checkpoint.pt")
    checkpoint_path = env_path("BIG_VAE_CHECKPOINT", raw_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BIG_VAE_CHECKPOINT does not exist: {checkpoint_path}")

    offline_root = env_path(
        "HELDOUT_ROOT",
        "post_train_research/big_vae_heldout_eval/artifacts/offline_dataset",
    )
    if not (offline_root / "manifest.json").exists():
        raise FileNotFoundError(f"Held-out offline dataset manifest not found: {offline_root / 'manifest.json'}")

    default_output = offline_root / "eval" / f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
    output_dir = env_path("EVAL_OUTPUT_DIR", default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = configure_process_logging(cfg=cfg, role="evaluate_big_vae_heldout", rank=0, force=True)
    LOGGER.info("Run log file: %s", log_path)
    LOGGER.info("Loading BigVAE checkpoint: %s", checkpoint_path)
    device = _resolve_eval_device(cfg)
    model, ckpt_cfg, ckpt_payload = _load_checkpoint_model(checkpoint_path, device)

    dataset = OfflineBigVAEDataset(
        root_dir=offline_root,
        shuffle_chunks=env_bool("EVAL_SHUFFLE_CHUNKS", False),
        shuffle_records_within_chunk=env_bool("EVAL_SHUFFLE_RECORDS_WITHIN_CHUNK", False),
        repeat=False,
        seed=env_int("EVAL_SEED", int(cfg.data.get("seed", 42))),
        weight_cache_size=env_int("EVAL_WEIGHT_CACHE_SIZE", 64),
        sampling_mode="balanced",
        sampling_group_keys=("dataset", "model"),
        sampling_window_size=env_int("EVAL_SAMPLING_WINDOW_SIZE_RECORDS", 2048),
        sampling_max_records_per_chunk_round=env_int("EVAL_MAX_RECORDS_PER_CHUNK_ROUND", 8),
        x_chunk_cache_size=env_int("EVAL_X_CHUNK_CACHE_SIZE", 4),
    )
    LOGGER.info("Held-out dataset summary: %s", dataset.summary())

    payload = _evaluate_dataset(
        model=model,
        ckpt_cfg=ckpt_cfg,
        dataset=dataset,
        device=device,
        output_dir=output_dir,
    )
    payload["checkpoint"] = {
        "path": str(checkpoint_path),
        "step": int(ckpt_payload.get("step", 0) or 0),
        "stage": int(ckpt_payload.get("stage", 0) or 0),
    }
    payload["offline_dataset"] = dataset.summary()

    write_json(output_dir / "metrics_summary.json", payload)
    write_json(output_dir / "metrics_global.json", payload["global"])
    write_json(output_dir / "metrics_macro.json", payload["macro"])
    write_csv(output_dir / "metrics_by_model.csv", payload["by_model"])
    write_csv(output_dir / "metrics_by_dataset.csv", payload["by_dataset"])
    write_csv(output_dir / "metrics_by_dataset_model_pair.csv", payload["by_dataset_model_pair"])
    write_csv(output_dir / "metrics_by_layer.csv", payload["by_layer"])
    LOGGER.info("Held-out eval complete: output_dir=%s global=%s macro=%s", output_dir, payload["global"], payload["macro"])


if __name__ == "__main__":
    main()
