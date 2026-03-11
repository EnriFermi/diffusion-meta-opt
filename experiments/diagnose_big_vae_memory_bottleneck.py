from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset import data_pipeline, setup_logging
from experiments.diagnose_big_vae_identity import (
    EPS,
    FrozenTarget,
    _apply_cli_overrides,
    _bool_text,
    _build_model_state,
    _collect_first_nonfinite,
    _compute_curriculum_slice_sizes_local,
    _compute_grad_stats_local,
    _configure_reproducibility,
    _derive_frozen_target,
    _direction_metrics,
    _exact_dir_loss,
    _instantiate_model,
    _load_yaml_without_defaults,
    _next_valid_sample_deterministic,
    _plot_grouped_lines,
    _resolve_device,
    _safe_scalar,
    _tensor_debug_stats_local,
    _threshold_steps,
    _write_csv,
    _write_json,
    _sample_synthetic_deterministic,
)


@dataclass(frozen=True, slots=True)
class ModeSpec:
    key: str
    dir_name: str
    title: str
    description: str
    debug_decoder_kv_source: str = "latents"
    debug_query_hint: str = "none"
    debug_direct_from_encoder_tokens: bool = False


MODE_SPECS: tuple[ModeSpec, ...] = (
    ModeSpec(
        key="A",
        dir_name="mode_A_baseline",
        title="MODE A baseline current model",
        description="Current model on frozen pair with z_shortcut disabled.",
    ),
    ModeSpec(
        key="B",
        dir_name="mode_B_encoder_kv",
        title="MODE B decoder cross-attends to final encoder patch tokens",
        description="Remove latent-memory compression by using final encoder patch tokens as decoder KV.",
        debug_decoder_kv_source="encoder_patch_tokens",
    ),
    ModeSpec(
        key="C",
        dir_name="mode_C_aligned_hint",
        title="MODE C aligned encoder-token hint in decoder queries",
        description="Inject aligned encoder patch token into each decoder query while keeping latent-memory KV.",
        debug_query_hint="aligned_encoder_token",
    ),
    ModeSpec(
        key="D",
        dir_name="mode_D_direct_from_encoder",
        title="MODE D direct prediction from final encoder patch token",
        description="Bypass decoder and predict directions directly from final encoder patch tokens.",
        debug_direct_from_encoder_tokens=True,
    ),
)


def _logger():
    import logging

    return logging.getLogger("diagnose_big_vae_memory_bottleneck")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compose_default_cfg() -> DictConfig:
    conf_root = _repo_root() / "conf"
    root_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "shared_runtime_environment" / "global_app_hydra_logging_and_model_registry.yaml"),
        _load_yaml_without_defaults(conf_root / "shared_runtime_environment" / "checkpoint_and_artifact_paths_for_mini_then_big.yaml"),
    )
    data_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "data_profiles" / "data_profile_common_settings.yaml"),
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "data_profiles" / "data_profile_for_big_vae_training.yaml"),
    )
    collector_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "collector_profiles" / "collector_common_settings.yaml"),
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "collector_profiles" / "collector_profile_async_gpu1.yaml"),
    )
    streaming_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "streaming_profiles" / "streaming_common_settings.yaml"),
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "streaming_profiles" / "streaming_profile_none.yaml"),
    )
    train_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "shared_training_parameters" / "trainer_execution_common.yaml"),
        _load_yaml_without_defaults(conf_root / "big_vae_experiment" / "trainer_parameters_specific_to_big_vae_stage.yaml"),
    )
    model_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "shared_training_parameters" / "model_backbone_distribution_and_mini_vae.yaml"),
        _load_yaml_without_defaults(conf_root / "big_vae_experiment" / "model_parameters_specific_to_big_vae_stage.yaml"),
    )
    run_profile_cfg = _load_yaml_without_defaults(conf_root / "run_profiles" / "train_big_vae.yaml")
    diagnostics_cfg = _load_yaml_without_defaults(conf_root / "big_vae_experiment" / "diagnostics_memory_bottleneck.yaml")

    return OmegaConf.merge(
        root_cfg,
        OmegaConf.create(
            {
                "data": data_cfg,
                "collector": collector_cfg,
                "streaming": streaming_cfg,
                "train": train_cfg,
                "model": model_cfg,
            }
        ),
        run_profile_cfg,
        diagnostics_cfg,
    )


def _enforce_diagnostic_constraints(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.train.distributed = False
        cfg.train.num_gpus = 1
        cfg.train.amp = False
        cfg.train.compile = False
        cfg.train.compile_dynamic = False
        cfg.train.compile_disable_cudagraphs = False
        cfg.train.tf32 = False
        cfg.train.cudnn_benchmark = False
        cfg.train.weight_decay = 0.0
        cfg.train.grad_clip_norm = 0.0
        cfg.train.grad_clip_norm_by_part = {}
        cfg.train.behavioral_coef = 0.0
        cfg.train.structural_coef = 1.0
        cfg.train.slice_batch_size = 16
        cfg.train.fixed_training_batch.enabled = True
        cfg.train.synthetic_layer_source.enabled = False
        cfg.train.telemetry.grad_layer_monitor.enabled = False
        cfg.train.telemetry.comet.enabled = False
        cfg.train.struct_loss.lambda_dir = 1.0
        cfg.train.struct_loss.lambda_scale = 0.0
        cfg.train.struct_loss.lambda_rec = 0.0
        cfg.train.struct_loss.lambda_rel = 0.0
        cfg.model.distribution.dropout = 0.0
        cfg.model.mini_vae.dropout = 0.0
        cfg.model.big_vae.dropout = 0.0
        cfg.model.big_vae.use_latent_sampling = False


def _artifact_root(cfg: DictConfig) -> Path:
    base = _repo_root() / str(cfg.diagnostics.get("output_root", "artifacts/big_vae_memory_bottleneck_diagnostics"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = base / timestamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slim_source_meta(meta: dict[str, Any]) -> dict[str, Any]:
    sample_meta = meta.get("sample_meta")
    if isinstance(sample_meta, dict):
        image_meta = sample_meta.get("image_meta")
        if isinstance(image_meta, list):
            image_meta = image_meta[:2]
        sample_meta = {
            "model_run_id": sample_meta.get("model_run_id"),
            "xy_sampling_mode": sample_meta.get("xy_sampling_mode"),
            "image_meta": image_meta,
        }
    return {
        "model_name": meta.get("model_name"),
        "layer_name": meta.get("layer_name"),
        "original_x_shape": meta.get("original_x_shape"),
        "original_W_shape": meta.get("original_W_shape"),
        "x_row_indices": meta.get("x_row_indices"),
        "sample_meta": sample_meta,
    }


def _source_item_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    meta = item["meta"]
    sample_meta = meta.get("sample_meta")
    image_meta = None
    model_run_id = None
    if isinstance(sample_meta, dict):
        image_meta = sample_meta.get("image_meta")
        model_run_id = sample_meta.get("model_run_id")
    return (
        meta.get("model_name"),
        meta.get("layer_name"),
        model_run_id,
        str(image_meta),
        tuple(item["W"].shape),
        tuple(item["x"].shape),
    )


def _collect_source_batch(cfg: DictConfig, *, logger, artifact_root: Path) -> list[dict[str, Any]]:
    frozen_cfg = cfg.diagnostics.frozen_pair
    source_kind = str(frozen_cfg.get("source_kind", "real")).strip().lower()
    batch_size = max(2, int(frozen_cfg.get("source_batch_size", 16)))
    seed = int(cfg.diagnostics.get("seed", 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    max_x_rows = int(frozen_cfg.get("max_x_rows", cfg.train.get("max_x_rows", 0)))

    items: list[dict[str, Any]] = []
    if source_kind == "synthetic":
        for idx in range(batch_size):
            x_cpu, W_cpu, meta = _sample_synthetic_deterministic(cfg, generator=generator)
            items.append({"x": x_cpu, "W": W_cpu, "meta": {**meta, "sample_idx": idx}})
    else:
        with data_pipeline(
            cfg,
            logger=logger,
            emit_run_report=False,
            rank=0,
        ) as (dataset, _collector):
            dataset_iter = iter(dataset)
            for idx in range(batch_size):
                x_cpu, W_cpu, meta = _next_valid_sample_deterministic(
                    dataset_iter,
                    max_x_rows=max_x_rows,
                    generator=generator,
                    logger=logger,
                )
                meta["sample_idx"] = idx
                items.append({"x": x_cpu, "W": W_cpu, "meta": meta})

    batch_meta = []
    for idx, item in enumerate(items):
        batch_meta.append(
            {
                "sample_idx": idx,
                "identity": list(_source_item_identity(item)),
                "meta": _slim_source_meta(item["meta"]),
                "x_shape": list(item["x"].shape),
                "W_shape": list(item["W"].shape),
            }
        )
    _write_json(artifact_root / "source_batch_meta.json", {"source_kind": source_kind, "items": _safe_scalar(batch_meta)})
    return items


def _pair_candidate_score(item_a: dict[str, Any], item_b: dict[str, Any]) -> int:
    meta_a = item_a["meta"]
    meta_b = item_b["meta"]
    score = 0
    if meta_a.get("model_name") == meta_b.get("model_name"):
        score += 4
    if meta_a.get("layer_name") == meta_b.get("layer_name"):
        score += 8
    if tuple(item_a["W"].shape) == tuple(item_b["W"].shape):
        score += 16
    if tuple(item_a["x"].shape) == tuple(item_b["x"].shape):
        score += 8
    return score


def _select_different_pair(
    items: list[dict[str, Any]],
    *,
    patch_size: int,
    max_T_patches: int,
    max_d_out: int,
) -> tuple[int, int, dict[str, Any]]:
    best: tuple[int, int, int, dict[str, Any]] | None = None
    for idx_a in range(len(items)):
        for idx_b in range(idx_a + 1, len(items)):
            item_a = items[idx_a]
            item_b = items[idx_b]
            T_a = int(item_a["W"].shape[0] // patch_size)
            T_b = int(item_b["W"].shape[0] // patch_size)
            d_out_use = min(int(item_a["W"].shape[1]), int(item_b["W"].shape[1]), max_d_out)
            T_use = min(T_a, T_b, max_T_patches)
            if T_use <= 0 or d_out_use <= 0:
                continue
            tensors_differ = not torch.equal(item_a["W"], item_b["W"]) or not torch.equal(item_a["x"], item_b["x"])
            if not tensors_differ:
                continue
            pair_meta = {
                "T_total_a": T_a,
                "T_total_b": T_b,
                "T_use": int(T_use),
                "d_out_use": int(d_out_use),
                "score": int(_pair_candidate_score(item_a, item_b)),
            }
            candidate = (pair_meta["score"], idx_a, idx_b, pair_meta)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("Could not find two different source objects with at least one full patch and one output column")
    _, idx_a, idx_b, pair_meta = best
    return idx_a, idx_b, pair_meta


def _slice_source_item(
    item: dict[str, Any],
    *,
    patch_size: int,
    T_use: int,
    d_out_use: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    W = item["W"]
    x = item["x"]
    d_in, d_out = W.shape
    T_total = int(d_in // patch_size)
    if T_use <= 0 or T_use > T_total:
        raise ValueError(f"Invalid T_use={T_use} for source shape {tuple(W.shape)} and patch_size={patch_size}")
    if d_out_use <= 0 or d_out_use > d_out:
        raise ValueError(f"Invalid d_out_use={d_out_use} for source shape {tuple(W.shape)}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if T_use == T_total:
        patch_idx = torch.arange(T_total, dtype=torch.long)
    else:
        patch_idx = torch.randperm(T_total, generator=generator)[:T_use].sort().values
    offsets = torch.arange(patch_size, dtype=torch.long)
    row_idx = (patch_idx.unsqueeze(1) * patch_size + offsets.unsqueeze(0)).flatten()
    W_s = W.index_select(0, row_idx)
    x_s = x.index_select(1, row_idx)

    if d_out_use == d_out:
        col_idx = torch.arange(d_out, dtype=torch.long)
    else:
        col_idx = torch.randperm(d_out, generator=generator)[:d_out_use].sort().values
        W_s = W_s.index_select(1, col_idx)

    slice_meta = {
        "selected_patch_indices": patch_idx.tolist(),
        "selected_row_indices": row_idx.tolist(),
        "selected_col_indices": col_idx.tolist(),
        "source_W_shape": list(W.shape),
        "source_x_shape": list(x.shape),
        "sliced_W_shape": list(W_s.shape),
        "sliced_x_shape": list(x_s.shape),
        "source_meta": _slim_source_meta(item["meta"]),
    }
    return W_s.clone(), x_s.clone(), slice_meta


def _frozen_target_payload(target: FrozenTarget) -> dict[str, Any]:
    return {
        "W_s": target.W_s.detach().cpu(),
        "x_s": target.x_s.detach().cpu(),
        "X_patch_target": target.X_patch_target.detach().cpu(),
        "u_target": target.u_target.detach().cpu(),
        "r_target": target.r_target.detach().cpu(),
        "w_dir": target.w_dir.detach().cpu(),
        "meta": dict(target.meta),
    }


def _build_frozen_pairs(
    cfg: DictConfig,
    *,
    logger,
    artifact_root: Path,
) -> tuple[dict[str, FrozenTarget], dict[str, Any]]:
    frozen_cfg = cfg.diagnostics.frozen_pair
    patch_size = int(cfg.model.get("patch_size", 16))
    if bool(frozen_cfg.get("use_curriculum_slice", True)):
        max_T_patches, max_d_out = _compute_curriculum_slice_sizes_local(cfg)
    else:
        max_T_patches = int(frozen_cfg.get("max_T_patches", 1))
        max_d_out = int(frozen_cfg.get("max_d_out", 1))
    if frozen_cfg.get("max_T_patches") is not None:
        max_T_patches = int(frozen_cfg.get("max_T_patches"))
    if frozen_cfg.get("max_d_out") is not None:
        max_d_out = int(frozen_cfg.get("max_d_out"))

    source_items = _collect_source_batch(cfg, logger=logger, artifact_root=artifact_root)
    idx_a, idx_b, pair_meta = _select_different_pair(
        source_items,
        patch_size=patch_size,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
    )
    seed = int(cfg.diagnostics.get("seed", 0))
    T_use = int(pair_meta["T_use"])
    d_out_use = int(pair_meta["d_out_use"])

    W_a, x_a, slice_meta_a = _slice_source_item(
        source_items[idx_a],
        patch_size=patch_size,
        T_use=T_use,
        d_out_use=d_out_use,
        seed=seed + 101,
    )
    W_b, x_b, slice_meta_b = _slice_source_item(
        source_items[idx_b],
        patch_size=patch_size,
        T_use=T_use,
        d_out_use=d_out_use,
        seed=seed + 202,
    )

    W_diff = torch.stack([W_a, W_b], dim=0)
    x_diff = torch.stack([x_a, x_b], dim=0)
    W_dup = torch.stack([W_a, W_a.clone()], dim=0)
    x_dup = torch.stack([x_a, x_a.clone()], dim=0)

    different_pair = _derive_frozen_target(
        W_diff,
        x_diff,
        patch_size=patch_size,
        gamma=float(cfg.train.struct_loss.gamma),
    )
    duplicate_pair = _derive_frozen_target(
        W_dup,
        x_dup,
        patch_size=patch_size,
        gamma=float(cfg.train.struct_loss.gamma),
    )

    different_pair.meta["pair_variant"] = "different_pair"
    duplicate_pair.meta["pair_variant"] = "duplicate_pair"
    for target in (different_pair, duplicate_pair):
        target.meta["source_batch_size"] = int(len(source_items))
        target.meta["fixed_source_batch_size"] = int(cfg.train.get("slice_batch_size", 16))
        target.meta["selected_pair_indices"] = [int(idx_a), int(idx_b)]
        target.meta["selection_score"] = int(pair_meta["score"])
        target.meta["selection_patch_T_use"] = int(T_use)
        target.meta["selection_d_out_use"] = int(d_out_use)
        target.meta["source_kind"] = str(frozen_cfg.get("source_kind", "real"))
        target.meta["source_batch_identities"] = [list(_source_item_identity(item)) for item in source_items]
        target.meta["different_pair_slice_meta"] = [slice_meta_a, slice_meta_b]
        target.meta["duplicate_pair_slice_meta"] = [slice_meta_a, slice_meta_a]
        target.meta["source_pair_meta"] = {
            "sample_a": _slim_source_meta(source_items[idx_a]["meta"]),
            "sample_b": _slim_source_meta(source_items[idx_b]["meta"]),
        }

    dump_payload = {
        "different_pair": _frozen_target_payload(different_pair),
        "duplicate_pair": _frozen_target_payload(duplicate_pair),
        "selected_pair_indices": [int(idx_a), int(idx_b)],
        "selection_meta": pair_meta,
        "source_batch": [
            {
                "sample_idx": idx,
                "identity": list(_source_item_identity(item)),
                "meta": _slim_source_meta(item["meta"]),
                "x_shape": list(item["x"].shape),
                "W_shape": list(item["W"].shape),
            }
            for idx, item in enumerate(source_items)
        ],
    }
    dump_path = artifact_root / "frozen_pair_dump.pt"
    torch.save(dump_payload, dump_path)

    reloaded = torch.load(dump_path, map_location="cpu", weights_only=False)
    immutability = {
        "different_pair_W_equal": bool(torch.equal(reloaded["different_pair"]["W_s"], different_pair.W_s)),
        "different_pair_x_equal": bool(torch.equal(reloaded["different_pair"]["x_s"], different_pair.x_s)),
        "duplicate_pair_W_equal": bool(torch.equal(reloaded["duplicate_pair"]["W_s"], duplicate_pair.W_s)),
        "duplicate_pair_x_equal": bool(torch.equal(reloaded["duplicate_pair"]["x_s"], duplicate_pair.x_s)),
    }
    immutability["all_equal"] = all(immutability.values())

    pair_sanity = {
        "duplicate_w_samples_equal": bool(torch.equal(duplicate_pair.W_s[0], duplicate_pair.W_s[1])),
        "duplicate_x_samples_equal": bool(torch.equal(duplicate_pair.x_s[0], duplicate_pair.x_s[1])),
        "different_first_matches_duplicate_first_W": bool(torch.equal(different_pair.W_s[0], duplicate_pair.W_s[0])),
        "different_first_matches_duplicate_first_x": bool(torch.equal(different_pair.x_s[0], duplicate_pair.x_s[0])),
    }
    pair_sanity["all_ok"] = all(pair_sanity.values())
    if not pair_sanity["all_ok"]:
        raise RuntimeError(f"Frozen-pair construction invariant failed: {pair_sanity}")

    meta_payload = {
        "source_kind": str(frozen_cfg.get("source_kind", "real")),
        "source_batch_size": int(len(source_items)),
        "fixed_source_batch_size": int(cfg.train.get("slice_batch_size", 16)),
        "selected_pair_indices": [int(idx_a), int(idx_b)],
        "selection_meta": pair_meta,
        "pair_variants": {
            "different_pair": _safe_scalar(different_pair.meta),
            "duplicate_pair": _safe_scalar(duplicate_pair.meta),
        },
        "source_batch": dump_payload["source_batch"],
        "immutability": immutability,
        "pair_sanity": pair_sanity,
    }
    _write_json(artifact_root / "frozen_pair_meta.json", _safe_scalar(meta_payload))
    return {
        "different_pair": different_pair,
        "duplicate_pair": duplicate_pair,
    }, meta_payload


def _mode_rows_to_loss_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "variant": row["variant"],
            "step": row["step"],
            "loss": row["loss"],
            "direction_pre_norm_mean": row["direction_pre_norm_mean"],
            "direction_pre_norm_std": row["direction_pre_norm_std"],
        }
        for row in rows
    ]


def _mode_rows_to_cosine_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "variant": row["variant"],
            "step": row["step"],
            "mean_cos": row["mean_cos"],
            "weighted_cos": row["weighted_cos"],
            "min_cos": row["min_cos"],
            "max_cos": row["max_cos"],
        }
        for row in rows
    ]


def _samplewise_dir_loss(frozen: FrozenTarget, pred_dirs: torch.Tensor) -> list[float]:
    T_loss = int(frozen.meta["used_T_in_loss"])
    pred_dirs_used = pred_dirs[:, :, :T_loss, :]
    cos = (pred_dirs_used * frozen.u_target).sum(dim=-1)
    per_output = (frozen.w_dir * (1.0 - cos)).sum(dim=-1)
    per_sample = per_output.mean(dim=-1)
    return [float(value.item()) for value in per_sample]


def _mode_short_interpretation(mode_metrics: dict[str, Any], threshold: float, ratio_alert: float) -> str:
    dup_best = float(mode_metrics["variants"]["duplicate_pair"]["best_loss"])
    diff_best = float(mode_metrics["variants"]["different_pair"]["best_loss"])
    ratio = float(mode_metrics["comparison"]["different_to_duplicate_ratio"])
    if dup_best > threshold:
        return "duplicate_pair failed; debug branch is suspect"
    if diff_best <= threshold:
        return "different_pair converged; two-point separation succeeded"
    if ratio >= ratio_alert:
        return "duplicate_pair converged but different_pair stayed high"
    return "different_pair still failed without a strong duplicate-vs-different separation"


def _run_mode_variant(
    frozen_cpu: FrozenTarget,
    *,
    cfg: DictConfig,
    device: torch.device,
    model_cfg: Any,
    base_state: dict[str, torch.Tensor],
    mode_spec: ModeSpec,
    variant_name: str,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logger = _logger()
    run_cfg = cfg.diagnostics.training
    steps = max(1, int(run_cfg.get("steps", 500)))
    lr = float(run_cfg.get("lr", cfg.train.get("lr", 5e-4)))

    frozen = frozen_cpu.to(device)
    model = _instantiate_model(model_cfg, base_state, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    rows: list[dict[str, Any]] = []
    best_loss = math.inf
    first_nonfinite: dict[str, Any] | None = None
    best_payload: dict[str, Any] = {}
    debug_shapes: dict[str, Any] | None = None

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        W_hat, _z, _logvar, pred_dirs, direction_pre_norms, debug_info = model.forward_debug(
            frozen.W_s,
            frozen.x_s,
            return_direction_pre_norms=True,
            disable_z_shortcut=True,
            debug_decoder_kv_source=mode_spec.debug_decoder_kv_source,
            debug_query_hint=mode_spec.debug_query_hint,
            debug_direct_from_encoder_tokens=mode_spec.debug_direct_from_encoder_tokens,
        )
        loss, _details = _exact_dir_loss(frozen, pred_dirs, W_hat=W_hat)
        direction_metrics = _direction_metrics(frozen, pred_dirs)
        direction_pre_norm_stats = _tensor_debug_stats_local(direction_pre_norms)
        sample_losses = _samplewise_dir_loss(frozen, pred_dirs)
        row = {
            "mode": mode_spec.key,
            "variant": variant_name,
            "step": int(step),
            "loss": float(loss.detach().item()),
            **direction_metrics,
            "direction_pre_norm_mean": float(direction_pre_norm_stats.get("mean", 0.0)),
            "direction_pre_norm_std": float(direction_pre_norm_stats.get("std", 0.0)),
            "direction_pre_norm_min": float(direction_pre_norm_stats.get("min", 0.0)),
            "direction_pre_norm_max": float(direction_pre_norm_stats.get("max", 0.0)),
        }
        for sample_idx, sample_loss in enumerate(sample_losses):
            row[f"sample_{sample_idx}_loss"] = float(sample_loss)

        if debug_shapes is None:
            debug_shapes = {
                "encoder_patch_tokens_shape": list(debug_info["encoder_patch_tokens"].shape),
                "decoder_queries_shape": list(debug_info["decoder_queries_init"].shape),
                "pred_dirs_shape": list(pred_dirs.shape),
                "decoder_kv_shape": None
                if debug_info.get("decoder_kv") is None
                else list(debug_info["decoder_kv"].shape),
                "T_forward": int(debug_info["T"]),
                "T_loss": int(frozen.meta["T_loss"]),
                "used_rows_in_loss": int(frozen.meta["used_rows_in_loss"]),
                "B": int(frozen.meta["B"]),
                "d_in": int(frozen.meta["d_in"]),
                "d_out": int(frozen.meta["d_out"]),
                "patch_size": int(frozen.meta["patch_size"]),
            }

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_payload = {
                "W_hat": W_hat.detach().cpu().clone(),
                "pred_dirs": pred_dirs.detach().cpu().clone(),
                "direction_pre_norms": direction_pre_norms.detach().cpu().clone(),
                "debug_shapes": debug_shapes,
            }

        if step < steps:
            loss.backward()
            grad_stats = _compute_grad_stats_local(model)
            row.update(
                {
                    "grad_global_norm": float(grad_stats.get("grad/global_norm", 0.0)),
                    "grad_rms": float(grad_stats.get("grad/rms", 0.0)),
                    "grad_distribution_encoder_rms": float(grad_stats.get("grad/distribution_encoder_rms", 0.0)),
                    "grad_patch_tokenizer_rms": float(grad_stats.get("grad/patch_tokenizer_rms", 0.0)),
                    "grad_encoder_rms": float(grad_stats.get("grad/encoder_rms", 0.0)),
                    "grad_decoder_rms": float(grad_stats.get("grad/decoder_rms", 0.0)),
                    "grad_big_vae_other_rms": float(grad_stats.get("grad/big_vae_other_rms", 0.0)),
                }
            )
            if first_nonfinite is None:
                first_nonfinite = _collect_first_nonfinite(
                    step=step,
                    loss=loss,
                    model=model,
                    W_hat=W_hat,
                    pred_dirs=pred_dirs,
                    direction_pre_norms=direction_pre_norms,
                )
            optimizer.step()
        else:
            row["grad_global_norm"] = math.nan
            row["grad_rms"] = math.nan

        rows.append(row)

    torch.save(best_payload, artifact_dir / f"best_bundle_{variant_name}.pt")
    summary = {
        "initial_loss": float(rows[0]["loss"]),
        "final_loss": float(rows[-1]["loss"]),
        "best_loss": float(min(float(row["loss"]) for row in rows)),
        "best_mean_cos": float(max(float(row["mean_cos"]) for row in rows)),
        "best_weighted_cos": float(max(float(row["weighted_cos"]) for row in rows)),
        "final_mean_cos": float(rows[-1]["mean_cos"]),
        "final_weighted_cos": float(rows[-1]["weighted_cos"]),
        "steps_to_threshold": _threshold_steps(rows),
        "first_nonfinite": _safe_scalar(first_nonfinite),
        "debug_shapes": debug_shapes,
        "frozen_meta": _safe_scalar(frozen.meta),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(
        "Mode %s [%s]: best_loss=%.6e final_loss=%.6e",
        mode_spec.key,
        variant_name,
        summary["best_loss"],
        summary["final_loss"],
    )
    return rows, summary


def _run_mode(
    mode_spec: ModeSpec,
    frozen_variants: dict[str, FrozenTarget],
    *,
    cfg: DictConfig,
    device: torch.device,
    model_cfg: Any,
    base_state: dict[str, torch.Tensor],
    artifact_root: Path,
) -> dict[str, Any]:
    mode_dir = artifact_root / mode_spec.dir_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    variants_summary: dict[str, Any] = {}
    for variant_name in ("duplicate_pair", "different_pair"):
        rows, summary = _run_mode_variant(
            frozen_variants[variant_name],
            cfg=cfg,
            device=device,
            model_cfg=model_cfg,
            base_state=base_state,
            mode_spec=mode_spec,
            variant_name=variant_name,
            artifact_dir=mode_dir,
        )
        all_rows.extend(rows)
        variants_summary[variant_name] = summary

    threshold = float(cfg.diagnostics.reporting.get("convergence_threshold", 1e-2))
    ratio_eps = float(cfg.diagnostics.reporting.get("ratio_eps", 1e-12))
    duplicate_best = float(variants_summary["duplicate_pair"]["best_loss"])
    different_best = float(variants_summary["different_pair"]["best_loss"])
    comparison = {
        "duplicate_pair_best_loss": duplicate_best,
        "different_pair_best_loss": different_best,
        "different_to_duplicate_ratio": different_best / max(duplicate_best, ratio_eps),
        "duplicate_pair_threshold_reached": bool(duplicate_best <= threshold),
        "different_pair_threshold_reached": bool(different_best <= threshold),
    }

    loss_rows = _mode_rows_to_loss_csv(all_rows)
    cosine_rows = _mode_rows_to_cosine_csv(all_rows)
    _write_csv(mode_dir / "loss.csv", _safe_scalar(loss_rows))
    _write_csv(mode_dir / "cosine.csv", _safe_scalar(cosine_rows))
    loss_plot_note = _plot_grouped_lines(
        all_rows,
        x_key="step",
        y_key="loss",
        group_key="variant",
        path=mode_dir / "loss_vs_step.png",
        title=mode_spec.title,
        ylabel="loss",
        ylog=True,
    )
    cosine_plot_note = _plot_grouped_lines(
        all_rows,
        x_key="step",
        y_key="mean_cos",
        group_key="variant",
        path=mode_dir / "cosine_vs_step.png",
        title=mode_spec.title,
        ylabel="mean_cos",
        ylog=False,
    )

    mode_metrics = {
        "mode": mode_spec.key,
        "title": mode_spec.title,
        "description": mode_spec.description,
        "debug_decoder_kv_source": mode_spec.debug_decoder_kv_source,
        "debug_query_hint": mode_spec.debug_query_hint,
        "debug_direct_from_encoder_tokens": bool(mode_spec.debug_direct_from_encoder_tokens),
        "variants": variants_summary,
        "comparison": comparison,
        "loss_plot_note": loss_plot_note,
        "cosine_plot_note": cosine_plot_note,
    }
    mode_metrics["short_interpretation"] = _mode_short_interpretation(
        mode_metrics,
        threshold=threshold,
        ratio_alert=float(cfg.diagnostics.reporting.get("ratio_alert", 5.0)),
    )
    _write_json(mode_dir / "metrics.json", _safe_scalar(mode_metrics))
    return mode_metrics


def _conclusion_from_modes(modes: dict[str, dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    def _dup_good(key: str) -> bool:
        return float(modes[key]["variants"]["duplicate_pair"]["best_loss"]) <= threshold

    def _diff_good(key: str) -> bool:
        return float(modes[key]["variants"]["different_pair"]["best_loss"]) <= threshold

    def _yes_no_inconclusive(valid: bool, value: bool) -> bool | None:
        return value if valid else None

    storage = _yes_no_inconclusive(
        _dup_good("A") and _dup_good("B"),
        (not _diff_good("A")) and _diff_good("B"),
    )
    if storage is False and _dup_good("A") and _dup_good("B") and _diff_good("A"):
        storage = False

    retrieval = _yes_no_inconclusive(
        _dup_good("B") and _dup_good("C"),
        (not _diff_good("B")) and _diff_good("C"),
    )
    if retrieval is False and _dup_good("B") and _dup_good("C") and _diff_good("B"):
        retrieval = False

    decoder = _yes_no_inconclusive(
        _dup_good("C") and _dup_good("D"),
        (not _diff_good("C")) and _diff_good("D"),
    )
    if decoder is False and _dup_good("C") and _dup_good("D") and _diff_good("C"):
        decoder = False

    encoder_tokens = _yes_no_inconclusive(
        _dup_good("D"),
        _diff_good("D"),
    )

    suspicious_modes = [
        key
        for key, payload in modes.items()
        if float(payload["variants"]["duplicate_pair"]["best_loss"]) > threshold
    ]
    return {
        "memory_storage_bottleneck": storage,
        "retrieval_bottleneck": retrieval,
        "decoder_bottleneck": decoder,
        "encoder_final_tokens_sufficiently_informative": encoder_tokens,
        "duplicate_pair_failures": suspicious_modes,
    }


def _validate_mode_consistency(modes: dict[str, dict[str, Any]], *, threshold: float, impossible_eps: float) -> None:
    violations: list[dict[str, Any]] = []
    for key, payload in modes.items():
        duplicate_best = float(payload["variants"]["duplicate_pair"]["best_loss"])
        different_best = float(payload["variants"]["different_pair"]["best_loss"])
        if different_best <= impossible_eps and duplicate_best > threshold:
            violations.append(
                {
                    "mode": key,
                    "duplicate_pair_best_loss": duplicate_best,
                    "different_pair_best_loss": different_best,
                }
            )
    if violations:
        raise RuntimeError(
            "Impossible diagnostics outcome detected: different_pair converged to ~0 while duplicate_pair did not. "
            f"Treat this run as invalid. Details: {violations}"
        )


def _comparison_rows(modes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("A", "B", "C", "D"):
        payload = modes[key]
        rows.append(
            {
                "mode": key,
                "title": payload["title"],
                "duplicate_pair_best_loss": payload["comparison"]["duplicate_pair_best_loss"],
                "different_pair_best_loss": payload["comparison"]["different_pair_best_loss"],
                "different_to_duplicate_ratio": payload["comparison"]["different_to_duplicate_ratio"],
                "duplicate_pair_threshold_reached": payload["comparison"]["duplicate_pair_threshold_reached"],
                "different_pair_threshold_reached": payload["comparison"]["different_pair_threshold_reached"],
                "short_interpretation": payload["short_interpretation"],
            }
        )
    return rows


def _write_summary_markdown(
    path: Path,
    *,
    checkpoint_path: str,
    frozen_meta: dict[str, Any],
    modes: dict[str, dict[str, Any]],
    conclusions: dict[str, Any],
    threshold: float,
) -> None:
    pair_meta = frozen_meta["pair_variants"]["different_pair"]
    table_lines = [
        "| Mode | Duplicate best | Different best | Different final | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key in ("A", "B", "C", "D"):
        payload = modes[key]
        table_lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    f"{payload['variants']['duplicate_pair']['best_loss']:.6e}",
                    f"{payload['variants']['different_pair']['best_loss']:.6e}",
                    f"{payload['variants']['different_pair']['final_loss']:.6e}",
                    payload["short_interpretation"],
                ]
            )
            + " |"
        )

    duplicate_lines = [
        f"- MODE {key}: best_loss=`{modes[key]['variants']['duplicate_pair']['best_loss']:.6e}` threshold=`{threshold:.2e}`"
        for key in ("A", "B", "C", "D")
    ]

    lines = [
        "# Big VAE Memory Bottleneck Diagnostics",
        "",
        f"- checkpoint: `{checkpoint_path or '<random_init>'}`",
        f"- frozen source batch size: `{frozen_meta['source_batch_size']}`",
        f"- selected pair indices: `{frozen_meta['selected_pair_indices']}`",
        f"- frozen pair description: `B={pair_meta['B']}`, `d_in={pair_meta['d_in']}`, `d_out={pair_meta['d_out']}`, `patch_size={pair_meta['patch_size']}`",
        f"- patch accounting: `T_forward={pair_meta['T_forward']}`, `T_loss={pair_meta['T_loss']}`, `used_rows_in_loss={pair_meta['used_rows_in_loss']}`",
        f"- selected source pair: `{pair_meta['source_pair_meta']['sample_a']['model_name']}` / `{pair_meta['source_pair_meta']['sample_b']['model_name']}`",
        "",
        "## Duplicate Sanity",
        "",
        *duplicate_lines,
        "",
        "## Modes A/B/C/D",
        "",
        *table_lines,
        "",
        "## Conclusions",
        "",
        f"- memory storage bottleneck: `{_bool_text(conclusions['memory_storage_bottleneck'])}`",
        f"- retrieval bottleneck: `{_bool_text(conclusions['retrieval_bottleneck'])}`",
        f"- decoder bottleneck: `{_bool_text(conclusions['decoder_bottleneck'])}`",
        f"- encoder final tokens sufficiently informative: `{_bool_text(conclusions['encoder_final_tokens_sufficiently_informative'])}`",
        f"- duplicate-pair implementation suspect modes: `{conclusions['duplicate_pair_failures']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(cfg: DictConfig) -> None:
    _enforce_diagnostic_constraints(cfg)
    artifact_root = _artifact_root(cfg)
    log_dir = artifact_root / "logs"
    cfg.logging.dir = str(log_dir)
    cfg.logging.file_name = "diagnose_big_vae_memory_bottleneck.log"
    cfg.logging.file_path = str(log_dir / cfg.logging.file_name)
    setup_logging(cfg, rank=0)
    logger = _logger()

    device = _resolve_device(cfg)
    _configure_reproducibility(int(cfg.diagnostics.get("seed", 0)), device)
    (artifact_root / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    logger.info(
        "Memory bottleneck diagnostics: device=%s distributed=%s amp=%s compile=%s source_batch_size=%s",
        device,
        cfg.train.get("distributed"),
        cfg.train.get("amp"),
        cfg.train.get("compile"),
        cfg.diagnostics.frozen_pair.get("source_batch_size"),
    )

    frozen_variants, frozen_meta = _build_frozen_pairs(cfg, logger=logger, artifact_root=artifact_root)
    model_cfg, base_state, checkpoint_path = _build_model_state(cfg, device, logger)

    mode_metrics: dict[str, dict[str, Any]] = {}
    for mode_spec in MODE_SPECS:
        mode_metrics[mode_spec.key] = _run_mode(
            mode_spec,
            frozen_variants,
            cfg=cfg,
            device=device,
            model_cfg=model_cfg,
            base_state=base_state,
            artifact_root=artifact_root,
        )

    threshold = float(cfg.diagnostics.reporting.get("convergence_threshold", 1e-2))
    conclusions = _conclusion_from_modes(mode_metrics, threshold=threshold)
    _validate_mode_consistency(
        mode_metrics,
        threshold=threshold,
        impossible_eps=float(cfg.diagnostics.reporting.get("impossible_pair_eps", 1e-6)),
    )
    comparison_rows = _comparison_rows(mode_metrics)
    _write_json(artifact_root / "comparison_table.json", {"rows": _safe_scalar(comparison_rows)})
    _write_csv(artifact_root / "comparison_table.csv", _safe_scalar(comparison_rows))

    summary = {
        "artifact_root": str(artifact_root),
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "frozen_pair_meta": frozen_meta,
        "modes": mode_metrics,
        "comparison_table": comparison_rows,
        "conclusions": conclusions,
    }
    _write_json(artifact_root / "summary.json", _safe_scalar(summary))
    _write_summary_markdown(
        artifact_root / "summary.md",
        checkpoint_path=checkpoint_path,
        frozen_meta=frozen_meta,
        modes=mode_metrics,
        conclusions=conclusions,
        threshold=threshold,
    )
    logger.info("Diagnostics completed. Report: %s", artifact_root / "summary.md")


def main(argv: list[str] | None = None) -> None:
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = _compose_default_cfg()
    cfg = _apply_cli_overrides(cfg, overrides)
    _run(cfg)


if __name__ == "__main__":
    main()
