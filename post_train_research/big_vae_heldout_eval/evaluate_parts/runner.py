from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
import torch
from omegaconf import DictConfig, open_dict

from big_vae.runtime.artifacts import write_artifact_layout
from ..common import (
    GroupedMetrics,
    apply_heldout_log_dir,
    coverage_report,
    default_heldout_eval_dir,
    default_heldout_root,
    env_bool,
    env_int,
    env_path,
    finite_metrics,
    primary_dataset_name,
    promote_run_profile_to_root,
    sanitize_programmatic_hydra_logging,
    write_csv,
    write_json,
)
from big_vae.datasets.offline import OfflineBigVAEDataset
from dataset.logging_utils import configure_process_logging
from training.big_vae.batch_debug import _compute_curriculum_slice_sizes, _stable_batch_shape_targets
from training.big_vae.runtime import _resolve_amp
from training.big_vae.source_batching import _build_training_batch_from_source_states
from training.big_vae.source_pool import _make_source_slice_state, _remaining_source_state_slices
from .core import (
    _compute_loss_metrics,
    _load_checkpoint_model,
    _record_metrics_writer,
    _resolve_eval_device,
    _source_record_from_sample,
)
from .decoder_adapter import IdentityDecoderAdapter, build_decoder_adapter_from_env
from .latent_dump import (
    _extract_latents_for_batch,
    _latent_dump_metadata_row,
    _select_balanced_latent_dump_entries,
    _store_latent_dump_entry,
    _write_latent_dump,
    plot_latent_dump,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("evaluate_big_vae_heldout")

def _evaluate_dataset(
    *,
    model: torch.nn.Module,
    ckpt_cfg: DictConfig,
    dataset: OfflineBigVAEDataset,
    device: torch.device,
    output_dir: Path,
    decoder_adapter: IdentityDecoderAdapter,
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
    latent_dump_enabled = env_bool("EVAL_LATENT_DUMP_ENABLED", True)
    latent_dump_max_entries = max(0, env_int("EVAL_LATENT_DUMP_MAX_ENTRIES", 256))
    latent_dump_max_slices_per_source = max(1, env_int("EVAL_LATENT_DUMP_MAX_SLICES_PER_SOURCE", 1))
    latent_dump_balance_enabled = env_bool("EVAL_LATENT_DUMP_BALANCE_ENABLED", True)
    latent_dump_balance_keys = _parse_latent_dump_balance_keys() if latent_dump_balance_enabled else ()
    default_max_per_group = 1 if latent_dump_balance_enabled else max(1, latent_dump_max_entries)
    latent_dump_max_per_group = max(1, env_int("EVAL_LATENT_DUMP_MAX_PER_GROUP", default_max_per_group))
    latent_dump_buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    latent_dump_seen_by_group: dict[tuple[str, ...], int] = {}
    latent_dump_rng = random.Random(seed)
    latent_dump_config = {
        "enabled": bool(latent_dump_enabled),
        "max_entries": int(latent_dump_max_entries),
        "max_slices_per_source": int(latent_dump_max_slices_per_source),
        "balance_enabled": bool(latent_dump_balance_enabled),
        "balance_keys": list(latent_dump_balance_keys),
        "max_per_group": int(latent_dump_max_per_group),
        "source": "posterior_mu_when_sampling_enabled_else_base_z",
        "point_unit": "slice",
    }

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
            dumped_for_source = 0
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
                W_batch = batch_payload.W.to(device=device, non_blocking=True)
                x_batch = batch_payload.x.to(device=device, non_blocking=True)
                x_mask_batch = batch_payload.x_mask.to(device=device, non_blocking=True)
                d_in_mask_batch = batch_payload.d_in_mask.to(device=device, non_blocking=True)
                d_out_mask_batch = batch_payload.d_out_mask.to(device=device, non_blocking=True)
                metrics = _compute_loss_metrics(
                    model=model,
                    cfg=ckpt_cfg,
                    W_s=W_batch,
                    x_s=x_batch,
                    x_mask_s=x_mask_batch,
                    d_in_mask_s=d_in_mask_batch,
                    d_out_mask_s=d_out_mask_batch,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    decoder_adapter=decoder_adapter,
                )
                source_dump_remaining = latent_dump_max_slices_per_source - dumped_for_source
                if latent_dump_enabled and latent_dump_max_entries > 0 and source_dump_remaining > 0:
                    candidate_take = min(int(source_dump_remaining), int(batch_slices))
                    for local_idx in range(candidate_take):
                        row = _latent_dump_metadata_row(
                            entry_index=0,
                            record_index=records_seen,
                            slice_index=evaluated_for_source + local_idx,
                            source_batch_index=source_batch_idx,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            layer_name=layer_name,
                            source=source,
                        )
                        group_key = _latent_dump_group_key(row, latent_dump_balance_keys)
                        row["balance_group"] = " | ".join(group_key)
                        replacement_idx = _latent_dump_replacement_index(
                            buckets=latent_dump_buckets,
                            seen_by_group=latent_dump_seen_by_group,
                            group_key=group_key,
                            max_per_group=latent_dump_max_per_group,
                            rng=latent_dump_rng,
                        )
                        if replacement_idx is None:
                            continue
                        mu_cpu, logvar_cpu = _extract_latents_for_batch(
                            model=model,
                            W_s=W_batch[local_idx : local_idx + 1],
                            x_s=x_batch[local_idx : local_idx + 1],
                            x_mask_s=x_mask_batch[local_idx : local_idx + 1],
                            d_in_mask_s=d_in_mask_batch[local_idx : local_idx + 1],
                            d_out_mask_s=d_out_mask_batch[local_idx : local_idx + 1],
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                        )
                        _store_latent_dump_entry(
                            buckets=latent_dump_buckets,
                            group_key=group_key,
                            replacement_idx=int(replacement_idx),
                            row=row,
                            latent=mu_cpu[:1],
                            logvar=logvar_cpu[:1],
                        )
                    dumped_for_source += int(candidate_take)
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
    payload["coverage"] = coverage_report(
        observed_datasets=grouped.by_dataset.keys(),
        observed_models=grouped.by_model.keys(),
        observed_pairs=grouped.by_pair.keys(),
        skipped=skipped,
    )
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
        "decoder_adapter": decoder_adapter.metadata(),
    }
    candidate_entries = sum(len(bucket) for bucket in latent_dump_buckets.values())
    latent_dump_config["candidate_groups"] = int(len(latent_dump_buckets))
    latent_dump_config["candidate_entries_before_global_cap"] = int(candidate_entries)
    selected_latent_entries = _select_balanced_latent_dump_entries(
        buckets=latent_dump_buckets,
        max_entries=latent_dump_max_entries,
        balance_keys=latent_dump_balance_keys,
        seed=seed,
    )
    latent_dump_rows = []
    latent_dump_latents = []
    latent_dump_logvars = []
    for entry_index, entry in enumerate(selected_latent_entries):
        row = dict(entry["row"])
        row["entry_index"] = int(entry_index)
        latent_dump_rows.append(row)
        latent_dump_latents.append(entry["latent"])
        latent_dump_logvars.append(entry["logvar"])
    payload["latent_dump"] = _write_latent_dump(
        output_dir=output_dir,
        latents=latent_dump_latents,
        logvars=latent_dump_logvars,
        rows=latent_dump_rows,
        config=latent_dump_config,
    )
    return payload


def _load_base_config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        return compose(config_name="big_vae/train/default")


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
        default_heldout_root(),
    )
    log_dir = apply_heldout_log_dir(cfg, root_dir=offline_root)
    if not (offline_root / "manifest.json").exists():
        raise FileNotFoundError(f"Held-out offline dataset manifest not found: {offline_root / 'manifest.json'}")

    default_output = default_heldout_eval_dir() / "runs" / f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
    output_dir = env_path("EVAL_OUTPUT_DIR", default_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_layout(
        output_dir,
        kind="post_train.heldout_eval",
        run_id=output_dir.name,
        files={
            "summary": output_dir / "metrics_summary.json",
            "record_metrics": output_dir / "record_metrics.csv",
            "latent_dump": output_dir / "latent_dump.pt",
        },
        dirs={"heldout_dataset": offline_root, "logs": log_dir, "latent_plots": output_dir / "latent_plots"},
        metadata={"checkpoint": checkpoint_path, "heldout_root": offline_root},
    )

    log_path = configure_process_logging(cfg=cfg, role="evaluate_big_vae_heldout", rank=0, force=True)
    LOGGER.info("Run log file: %s", log_path)
    LOGGER.info("Held-out log dir: %s", log_dir)
    LOGGER.info("Loading BigVAE checkpoint: %s", checkpoint_path)
    device = _resolve_eval_device(cfg)
    model, ckpt_cfg, ckpt_payload = _load_checkpoint_model(checkpoint_path, device)
    decoder_adapter = build_decoder_adapter_from_env(
        model=model,
        big_vae_checkpoint_path=checkpoint_path,
        device=device,
        logger=LOGGER,
    )

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
        decoder_adapter=decoder_adapter,
    )
    payload["decoder_adapter"] = decoder_adapter.metadata()
    payload["checkpoint"] = {
        "path": str(checkpoint_path),
        "step": int(ckpt_payload.get("step", 0) or 0),
        "stage": int(ckpt_payload.get("stage", 0) or 0),
    }
    payload["offline_dataset"] = dataset.summary()
    latent_dump_path = str(payload.get("latent_dump", {}).get("path", "")).strip()
    if latent_dump_path and env_bool("EVAL_LATENT_PLOT_ENABLED", True):
        payload["latent_plots"] = plot_latent_dump(
            latent_dump_path,
            output_dir=env_path("EVAL_LATENT_PLOT_DIR", output_dir / "latent_plots"),
            seed=env_int("EVAL_SEED", int(cfg.data.get("seed", 42))),
        )
    else:
        payload["latent_plots"] = {"enabled": False}

    write_json(output_dir / "metrics_summary.json", payload)
    write_json(output_dir / "coverage.json", payload["coverage"])
    write_json(output_dir / "metrics_global.json", payload["global"])
    write_json(output_dir / "metrics_macro.json", payload["macro"])
    write_csv(output_dir / "metrics_by_model.csv", payload["by_model"])
    write_csv(output_dir / "metrics_by_dataset.csv", payload["by_dataset"])
    write_csv(output_dir / "metrics_by_dataset_model_pair.csv", payload["by_dataset_model_pair"])
    write_csv(output_dir / "metrics_by_layer.csv", payload["by_layer"])
    LOGGER.info(
        "Held-out eval complete: output_dir=%s coverage_ok=%s global=%s macro=%s",
        output_dir,
        bool(payload["coverage"].get("ok", False)),
        payload["global"],
        payload["macro"],
    )
    if not bool(payload["coverage"].get("ok", False)):
        LOGGER.error("Held-out coverage check failed: %s", payload["coverage"])
        if env_bool("EVAL_REQUIRE_FULL_COVERAGE", env_int("EVAL_MAX_RECORDS", 0) <= 0):
            raise RuntimeError(f"Held-out coverage check failed; see {output_dir / 'coverage.json'}")


if __name__ == "__main__":
    main()
