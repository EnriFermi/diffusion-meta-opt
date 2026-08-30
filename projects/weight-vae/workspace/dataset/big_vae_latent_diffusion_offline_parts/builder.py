from __future__ import annotations

import logging
import random
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from big_vae.datasets.offline import infer_layer_depth, infer_layer_type
from dataset.shared.types import SharedSample
from training.big_vae_latent_diffusion import (
    build_cond_global_from_dist_var_pooled,
    encode_big_vae_layer_batch,
)

from .source_slicing import (
    _LatentDiffusionSourceState,
    _allocate_capped_proportional_quotas,
    _build_latent_diffusion_batch_from_source_states,
    _consume_slice_from_latent_diffusion_source_state,
    _distribution_match_source_info,
    _latent_diffusion_source_states_total_remaining_slices,
    _make_latent_diffusion_source_state,
    _pad_source_samples,
    _prune_exhausted_latent_diffusion_source_states_with_offset,
    _resolve_distribution_match_cfg,
    _resolve_explicit_slice_target,
    _sample_slot_count_without_replacement,
    _source_key,
)
from .writer import BigVAELatentDiffusionOfflineWriter

def build_big_vae_latent_diffusion_offline_dataset(
    cfg: DictConfig,
    *,
    big_vae: Any,
    dataset_iter: Iterable[Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("latent_diffusion_dataset", {})
    if not isinstance(dataset_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset must be a mapping")
    builder_cfg = dataset_cfg.get("builder", {})
    if builder_cfg is None:
        builder_cfg = {}
    if not isinstance(builder_cfg, (dict, DictConfig)):
        raise TypeError("latent_diffusion_dataset.builder must be a mapping")

    root_dir = str(dataset_cfg.get("root_dir", "") or "").strip()
    if not root_dir:
        raise ValueError("latent_diffusion_dataset.root_dir must be set")
    if not bool(getattr(big_vae, "use_distribution_encoder", False)):
        raise ValueError("latent diffusion dataset build requires frozen BigVAE with distribution encoder enabled")
    target = _resolve_explicit_slice_target(builder_cfg, patch_size=int(big_vae.cfg.patch_size))

    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    writer = BigVAELatentDiffusionOfflineWriter(
        root_dir=root_dir,
        overwrite_existing=bool(builder_cfg.get("overwrite_existing", False)),
        chunk_size_records=int(builder_cfg.get("chunk_size_records", 128)),
        patch_size=int(target.patch_size),
        target_T_patches=int(target.target_T_patches),
        target_d_out=int(target.target_d_out),
        store_decoder_aux_tensors=bool(builder_cfg.get("store_decoder_aux_tensors", False)),
        logger=logger,
        config_snapshot=cfg_snapshot if isinstance(cfg_snapshot, dict) else {},
    )
    logger_local = logger or logging.getLogger("dataset.big_vae_latent_diffusion_offline")
    batch_size = max(1, int(builder_cfg.get("batch_size", 8)))
    encode_batch_size = max(1, int(builder_cfg.get("encode_batch_size", 1)))
    store_decoder_aux_tensors = bool(builder_cfg.get("store_decoder_aux_tensors", False))
    log_every_batches = max(1, int(builder_cfg.get("log_every_batches", 100)))
    max_records = max(0, int(builder_cfg.get("max_records", 0)))
    distribution_match_cfg = _resolve_distribution_match_cfg(builder_cfg)
    max_latent_records = int(distribution_match_cfg.max_latent_records)
    seed = int(builder_cfg.get("seed", dataset_cfg.get("source", {}).get("seed", 42)))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    sampling_rng = random.Random(seed)
    model_device = next(big_vae.parameters()).device

    pending: list[Any] = []
    total_seen = 0
    total_shape_incompatible = 0
    source_records_seen = 0
    compatible_source_records = 0
    available_latent_records = 0
    target_latent_records = 0

    def _latent_limit_reached() -> bool:
        return max_latent_records > 0 and int(writer._accepted_records) >= max_latent_records

    def _encode_and_ingest_batch_payload(batch_payload: Mapping[str, Any]) -> int:
        if _latent_limit_reached():
            return 0
        with torch.no_grad():
            encoded = encode_big_vae_layer_batch(
                big_vae,
                W=batch_payload["W"].to(device=model_device),
                X=batch_payload["X"].to(device=model_device),
                x_mask=batch_payload["x_mask"].to(device=model_device),
                d_in_mask=batch_payload["d_in_mask"].to(device=model_device),
                d_out_mask=batch_payload["d_out_mask"].to(device=model_device),
            )
        cond_patch = encoded["cond_patch"]
        dist_var_pooled = encoded.get("dist_var_pooled")
        patch_mask = encoded["patch_mask"]
        latent_mu = encoded["latent_mu"]
        latent_logvar = encoded["latent_logvar"]
        if not torch.is_tensor(cond_patch) or not torch.is_tensor(patch_mask):
            raise RuntimeError("BigVAE latent diffusion build expected tensor cond_patch and patch_mask")
        if not torch.is_tensor(latent_mu) or not torch.is_tensor(latent_logvar):
            raise RuntimeError("BigVAE latent diffusion build expected tensor latent_mu and latent_logvar")

        batch = int(latent_mu.shape[0])
        remaining_emit = batch
        if max_latent_records > 0:
            remaining_emit = min(batch, max(0, max_latent_records - int(writer._accepted_records)))
        if remaining_emit <= 0:
            return 0

        for idx in range(remaining_emit):
            valid_t = int(patch_mask[idx].to(dtype=torch.long).sum().item())
            valid_rows = int(batch_payload["x_mask"][idx].to(dtype=torch.long).sum().item())
            valid_d_in = int(batch_payload["d_in_mask"][idx].to(dtype=torch.long).sum().item())
            valid_d_out = int(batch_payload["d_out_mask"][idx].to(dtype=torch.long).sum().item())
            cond_global = None
            if torch.is_tensor(dist_var_pooled):
                cond_global_full = build_cond_global_from_dist_var_pooled(
                    dist_var_pooled=dist_var_pooled[idx : idx + 1],
                    patch_mask=patch_mask[idx : idx + 1],
                )
                if cond_global_full is not None:
                    cond_global = cond_global_full[0]
            record = {
                "source_key": _source_key(batch_payload["model_names"][idx], batch_payload["layer_names"][idx]),
                "model_name": batch_payload["model_names"][idx],
                "layer_name": batch_payload["layer_names"][idx],
                "layer_type": infer_layer_type(batch_payload["layer_names"][idx]),
                "layer_depth": infer_layer_depth(batch_payload["layer_names"][idx]),
                "d_in": valid_d_in,
                "d_out": valid_d_out,
                "cond_patch": cond_patch[idx, :valid_t],
                "patch_mask": torch.ones(valid_t, dtype=torch.bool),
                "cond_global": cond_global,
                "latent_mu": latent_mu[idx],
                "latent_logvar": latent_logvar[idx],
                "meta": batch_payload["meta"][idx],
                "target_T_patches": int(target.target_T_patches),
                "target_d_out": int(target.target_d_out),
            }
            if store_decoder_aux_tensors:
                record["X"] = batch_payload["X"][idx, :valid_rows, :valid_d_in]
                record["W"] = batch_payload["W"][idx, :valid_d_in, :valid_d_out]
                record["x_mask"] = batch_payload["x_mask"][idx, :valid_rows]
                record["d_in_mask"] = batch_payload["d_in_mask"][idx, :valid_d_in]
                record["d_out_mask"] = batch_payload["d_out_mask"][idx, :valid_d_out]
            writer.ingest(record)
        return int(remaining_emit)

    def _encode_and_ingest_samples(samples: Sequence[Any]) -> int:
        if not samples or _latent_limit_reached():
            return 0
        return _encode_and_ingest_batch_payload(_pad_source_samples(samples))

    def _flush_pending_exhaustive() -> None:
        nonlocal pending, total_shape_incompatible
        if not pending:
            return
        source_states: list[_LatentDiffusionSourceState] = []
        for raw_sample in pending:
            state = _make_latent_diffusion_source_state(raw_sample, target=target, rng=rng)
            if state is None:
                total_shape_incompatible += 1
                continue
            source_states.append(state)
        if not source_states:
            pending = []
            return
        start_offset = 0
        while source_states and not _latent_limit_reached():
            emit_batch_size = min(
                int(encode_batch_size),
                int(_latent_diffusion_source_states_total_remaining_slices(source_states)),
            )
            if max_latent_records > 0:
                emit_batch_size = min(emit_batch_size, max(0, max_latent_records - int(writer._accepted_records)))
            if emit_batch_size <= 0:
                break
            batch_payload, start_offset, _used_source_indices = _build_latent_diffusion_batch_from_source_states(
                source_states,
                batch_size=emit_batch_size,
                start_offset=start_offset,
            )
            _encode_and_ingest_batch_payload(batch_payload)
            source_states, start_offset = _prune_exhausted_latent_diffusion_source_states_with_offset(
                list(source_states),
                start_offset,
            )
        pending = []

    if distribution_match_cfg.mode == "source_proportional_joint":
        if iter(dataset_iter) is dataset_iter:
            raise TypeError(
                "source_proportional_joint requires a reiterable dataset object, not a one-shot iterator"
            )

        group_source_counts: OrderedDict[tuple[str, ...], int] = OrderedDict()
        group_capacity_totals: OrderedDict[tuple[str, ...], int] = OrderedDict()
        scan_log_every = max(1, batch_size * log_every_batches)
        for sample in dataset_iter:
            total_seen += 1
            info = _distribution_match_source_info(
                sample,
                target=target,
                group_keys=distribution_match_cfg.group_keys,
                max_slices_per_source_record=distribution_match_cfg.max_slices_per_source_record,
            )
            if info is None:
                total_shape_incompatible += 1
            else:
                group_key, effective_capacity = info
                compatible_source_records += 1
                group_source_counts[group_key] = int(group_source_counts.get(group_key, 0)) + 1
                group_capacity_totals[group_key] = int(group_capacity_totals.get(group_key, 0)) + int(effective_capacity)
            if total_seen % scan_log_every == 0:
                logger_local.info(
                    "Latent diffusion dataset prescan: seen=%s compatible=%s skipped_shape_incompatible=%s "
                    "groups=%s mode=%s root=%s",
                    int(total_seen),
                    int(compatible_source_records),
                    int(total_shape_incompatible),
                    int(len(group_source_counts)),
                    distribution_match_cfg.mode,
                    root_dir,
                )
            if max_records > 0 and total_seen >= max_records:
                break

        source_records_seen = int(total_seen)
        available_latent_records = int(sum(int(value) for value in group_capacity_totals.values()))
        target_latent_records = int(available_latent_records)
        if max_latent_records > 0:
            target_latent_records = min(int(target_latent_records), int(max_latent_records))
        if target_latent_records <= 0:
            raise RuntimeError(
                "Latent diffusion dataset prescan found no compatible latent records under the requested shape/limit: "
                f"target_T_patches={target.target_T_patches} target_d_out={target.target_d_out} "
                f"max_latent_records={max_latent_records}"
            )

        group_quota_totals = _allocate_capped_proportional_quotas(
            weights=group_source_counts,
            capacities=group_capacity_totals,
            total_target=target_latent_records,
        )
        group_quota_remaining = {group_key: int(value) for group_key, value in group_quota_totals.items()}
        group_capacity_remaining = {group_key: int(value) for group_key, value in group_capacity_totals.items()}
        selected_pending: list[SharedSample] = []
        selected_source_records = 0
        second_pass_seen = 0

        def _flush_selected_pending(*, force: bool = False) -> None:
            nonlocal selected_pending
            while selected_pending and (force or len(selected_pending) >= encode_batch_size) and not _latent_limit_reached():
                current_batch = selected_pending[:encode_batch_size]
                del selected_pending[: len(current_batch)]
                _encode_and_ingest_samples(current_batch)
            if _latent_limit_reached():
                selected_pending = []

        for sample in dataset_iter:
            second_pass_seen += 1
            info = _distribution_match_source_info(
                sample,
                target=target,
                group_keys=distribution_match_cfg.group_keys,
                max_slices_per_source_record=distribution_match_cfg.max_slices_per_source_record,
            )
            if info is not None:
                group_key, effective_capacity = info
                group_success_remaining = int(group_quota_remaining.get(group_key, 0))
                group_population_remaining = int(group_capacity_remaining.get(group_key, 0))
                take_n = _sample_slot_count_without_replacement(
                    num_slots=effective_capacity,
                    success_remaining=group_success_remaining,
                    population_remaining=group_population_remaining,
                    rng=sampling_rng,
                )
                group_quota_remaining[group_key] = max(0, group_success_remaining - int(take_n))
                group_capacity_remaining[group_key] = max(0, group_population_remaining - int(effective_capacity))
                if take_n > 0:
                    state = _make_latent_diffusion_source_state(sample, target=target, rng=rng)
                    if state is None:
                        raise RuntimeError(
                            "Latent diffusion source sample became incompatible between prescan and second pass"
                        )
                    selected_source_records += 1
                    for _ in range(int(take_n)):
                        sliced = _consume_slice_from_latent_diffusion_source_state(state)
                        if sliced is None:
                            raise RuntimeError(
                                "Latent diffusion source state exhausted before satisfying allocated slice quota"
                            )
                        selected_pending.append(sliced)
                    _flush_selected_pending()
            if second_pass_seen % scan_log_every == 0:
                logger_local.info(
                    "Latent diffusion dataset build progress: source_seen=%s accepted=%s target_latent_records=%s "
                    "selected_source_records=%s mode=%s root=%s",
                    int(second_pass_seen),
                    int(writer._accepted_records),
                    int(target_latent_records),
                    int(selected_source_records),
                    distribution_match_cfg.mode,
                    root_dir,
                )
            if _latent_limit_reached() or sum(group_quota_remaining.values()) <= 0:
                break
            if max_records > 0 and second_pass_seen >= max_records:
                break

        _flush_selected_pending(force=True)
        remaining_group_quota = int(sum(group_quota_remaining.values()))
        if remaining_group_quota > 0:
            logger_local.warning(
                "Latent diffusion proportional build finished with unsatisfied group quota: remaining=%s "
                "accepted=%s target_latent_records=%s root=%s",
                remaining_group_quota,
                int(writer._accepted_records),
                int(target_latent_records),
                root_dir,
            )
    else:
        for sample in dataset_iter:
            pending.append(sample)
            total_seen += 1
            if len(pending) >= batch_size:
                _flush_pending_exhaustive()
                processed_batches = total_seen // batch_size
                if processed_batches % log_every_batches == 0:
                    logger_local.info(
                        "Latent diffusion dataset build progress: seen=%s accepted=%s skipped_shape_incompatible=%s "
                        "target_T_patches=%s target_d_out=%s mode=%s root=%s",
                        total_seen,
                        int(writer._accepted_records),
                        int(total_shape_incompatible),
                        int(target.target_T_patches),
                        int(target.target_d_out),
                        distribution_match_cfg.mode,
                        root_dir,
                    )
            if _latent_limit_reached() or (max_records > 0 and total_seen >= max_records):
                break

        if not _latent_limit_reached():
            _flush_pending_exhaustive()
        source_records_seen = int(total_seen)

    summary = writer.close()
    summary["skipped_shape_incompatible"] = int(total_shape_incompatible)
    summary["source_records_seen"] = int(source_records_seen or total_seen)
    summary["distribution_match_mode"] = str(distribution_match_cfg.mode)
    summary["distribution_match_group_keys"] = list(distribution_match_cfg.group_keys)
    summary["max_latent_records"] = int(max_latent_records)
    summary["max_slices_per_source_record"] = int(distribution_match_cfg.max_slices_per_source_record)
    summary["compatible_source_records"] = int(compatible_source_records)
    summary["available_latent_records"] = int(available_latent_records)
    summary["target_latent_records"] = int(target_latent_records or summary.get("accepted_records", 0))
    logger_local.info(
        "Latent diffusion dataset build complete: root=%s accepted_records=%s skipped_shape_incompatible=%s "
        "source_records_seen=%s target_latent_records=%s target_T_patches=%s target_d_out=%s mode=%s actual_size_gb=%.2f",
        root_dir,
        int(summary.get("accepted_records", 0)),
        int(total_shape_incompatible),
        int(summary.get("source_records_seen", 0)),
        int(summary.get("target_latent_records", 0)),
        int(target.target_T_patches),
        int(target.target_d_out),
        distribution_match_cfg.mode,
        float(summary.get("actual_size_gb", 0.0)),
    )
    return summary
