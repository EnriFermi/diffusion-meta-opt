from __future__ import annotations

import json
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ExperimentConfig, config_hash, config_to_dict, read_config, run_dir, torch_dtype, write_config
from .core import (
    decoder_jacobians,
    decode_weights,
    encode_weights,
    generate_weight_pool,
    geometry_metrics_from_jacobians,
    load_vision_tensors,
    random_near_identity_flow,
    train_posthoc_flow,
    train_weight_vae,
    atomic_torch_save,
    flow_training_cache_key,
    load_torch_cache,
    vae_training_cache_key,
    weight_pool_cache_key,
)
from .downstream import DownstreamContext, tune_and_evaluate_downstream
from .progress import make_progress


def _set_stage(progress, index: int, total: int, name: str) -> None:
    progress.set_description(f"pipeline {index}/{total} {name}")


@dataclass(slots=True)
class ExperimentTables:
    vae_metrics: pd.DataFrame
    geometry: pd.DataFrame
    selected_lrs: pd.DataFrame
    downstream_results: pd.DataFrame
    downstream_curves: pd.DataFrame
    output_dir: Path
    config: dict[str, Any]
    interpretation_markdown: str


def _read_existing(output_dir: Path) -> ExperimentTables:
    config_payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    interpretation_path = output_dir / "interpretation.md"
    return ExperimentTables(
        vae_metrics=pd.read_csv(output_dir / "vae_metrics.csv"),
        geometry=pd.read_csv(output_dir / "geometry.csv"),
        selected_lrs=pd.read_csv(output_dir / "selected_lrs.csv"),
        downstream_results=pd.read_csv(output_dir / "downstream_results.csv"),
        downstream_curves=pd.read_csv(output_dir / "downstream_curves.csv"),
        output_dir=output_dir,
        config=config_payload,
        interpretation_markdown=interpretation_path.read_text(encoding="utf-8") if interpretation_path.is_file() else "",
    )


def _all_outputs_exist(output_dir: Path, cfg: ExperimentConfig) -> bool:
    variants = _vae_variants_from_coeff(float(cfg.vae_geometry_reg_coeff))
    multi_variant = len(variants) > 1
    required = (
        "config.json",
        "weight_pool.pt",
        "vae_metrics.csv",
        "geometry.csv",
        "selected_lrs.csv",
        "downstream_results.csv",
        "downstream_curves.csv",
    )
    if not all((output_dir / name).is_file() for name in required):
        return False
    weight_payload = load_torch_cache(output_dir / "weight_pool.pt")
    if weight_payload is None:
        return False
    weight_key = weight_pool_cache_key(cfg)
    if weight_payload.get("cache_key") != weight_key:
        return False
    weights = weight_payload.get("weights")
    if not isinstance(weights, torch.Tensor):
        return False
    for variant, _reg_coeff in variants:
        variant_cfg = replace(cfg, vae_geometry_reg_coeff=float(_reg_coeff))
        vae_path = _variant_path(output_dir, "vae_checkpoint.pt", variant=variant, multi_variant=multi_variant)
        flow_path = _variant_path(output_dir, "flow_state.pt", variant=variant, multi_variant=multi_variant)
        random_flow_path = _variant_path(output_dir, "random_flow_state.pt", variant=variant, multi_variant=multi_variant)
        if not vae_path.is_file() or not flow_path.is_file() or not random_flow_path.is_file():
            return False
        vae_payload = load_torch_cache(vae_path)
        flow_payload = load_torch_cache(flow_path)
        random_flow_payload = load_torch_cache(random_flow_path)
        if vae_payload is None or flow_payload is None or random_flow_payload is None:
            return False
        vae_key = vae_training_cache_key(variant_cfg, weights, upstream_cache_key=weight_key)
        if vae_payload.get("cache_key") != vae_key:
            return False
        train_indices = vae_payload.get("train_indices")
        if not isinstance(train_indices, torch.Tensor):
            return False
        z_shape = (int(train_indices.numel()), int(variant_cfg.latent_dim))
        z_train_shape_only = torch.empty(z_shape)
        flow_key = flow_training_cache_key(variant_cfg, z_train_shape_only, upstream_cache_key=vae_key)
        if flow_payload.get("cache_key") != flow_key:
            return False
        if random_flow_payload.get("cache_key") != f"random-{flow_key}":
            return False
    return True


def _vae_variants_from_coeff(reg_coeff: float) -> tuple[tuple[str, float], ...]:
    if float(reg_coeff) > 0.0:
        return (("baseline", 0.0), ("regularized", float(reg_coeff)))
    return (("baseline", 0.0),)


def _variant_path(output_dir: Path, name: str, *, variant: str, multi_variant: bool) -> Path:
    if not multi_variant:
        return output_dir / name
    path = Path(name)
    return output_dir / f"{path.stem}_{variant}{path.suffix}"


def _vae_quality_rows(
    *,
    cfg: ExperimentConfig,
    trained,
    weights: torch.Tensor,
    spec,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
) -> pd.DataFrame:
    from .core import evaluate_flat_model

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    vae = trained.vae.to(device=device, dtype=dtype)
    weights_device = weights.to(device=device, dtype=dtype)
    rows: list[dict[str, float | int | str]] = []
    eval_indices = trained.val_indices[: min(16, int(trained.val_indices.numel()))]
    if int(eval_indices.numel()) == 0:
        eval_indices = trained.train_indices[: min(16, int(trained.train_indices.numel()))]
    progress = make_progress(cfg, total=int(eval_indices.numel()), desc="VAE quality eval")
    try:
        for local_idx, idx in enumerate(eval_indices.tolist()):
            w = weights_device[int(idx)]
            z = encode_weights(vae, trained.normalizer, w.reshape(1, -1)).squeeze(0)
            recon = decode_weights(vae, trained.normalizer, z.reshape(1, -1)).squeeze(0)
            z_roundtrip = encode_weights(vae, trained.normalizer, recon.reshape(1, -1)).squeeze(0)
            raw_train_loss, raw_train_acc = evaluate_flat_model(w, images=train_images, labels=train_labels, spec=spec)
            dec_train_loss, dec_train_acc = evaluate_flat_model(recon, images=train_images, labels=train_labels, spec=spec)
            raw_test_loss, raw_test_acc = evaluate_flat_model(w, images=test_images, labels=test_labels, spec=spec)
            dec_test_loss, dec_test_acc = evaluate_flat_model(recon, images=test_images, labels=test_labels, spec=spec)
            reconstruction_rel_l2 = float(((recon - w).float().norm() / w.float().norm().clamp_min(1e-12)).detach().cpu().item())
            rows.append(
                {
                    "sample_index": int(local_idx),
                    "weight_index": int(idx),
                    "reconstruction_mse": float((recon - w).square().mean().detach().cpu().item()),
                    "reconstruction_rel_l2": reconstruction_rel_l2,
                    "latent_roundtrip_l2": float((z_roundtrip - z).float().norm().detach().cpu().item()),
                    "raw_train_loss": raw_train_loss,
                    "raw_train_acc": raw_train_acc,
                    "decoded_train_loss": dec_train_loss,
                    "decoded_train_acc": dec_train_acc,
                    "raw_test_loss": raw_test_loss,
                    "raw_test_acc": raw_test_acc,
                    "decoded_test_loss": dec_test_loss,
                    "decoded_test_acc": dec_test_acc,
                }
            )
            progress.set_postfix({"sample": f"{local_idx + 1}/{int(eval_indices.numel())}", "rel_l2": f"{reconstruction_rel_l2:.3g}", "decoded_test_acc": f"{dec_test_acc:.3f}"})
            progress.update(1)
    finally:
        progress.close()
    train_history = trained.metrics.copy()
    train_history["sample_index"] = -1
    train_history["weight_index"] = -1
    train_history["record_type"] = "vae_train_history"
    quality = pd.DataFrame(rows)
    quality["record_type"] = "vae_quality"
    return pd.concat([train_history, quality], ignore_index=True, sort=False)


def _geometry_table(
    *,
    cfg: ExperimentConfig,
    trained,
    weights: torch.Tensor,
    trained_flow: torch.nn.Module,
    random_flow: torch.nn.Module,
) -> pd.DataFrame:
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    vae = trained.vae.to(device=device, dtype=dtype).eval()
    weights_device = weights.to(device=device, dtype=dtype)
    eval_indices = trained.val_indices[: min(int(cfg.geometry_eval_samples), int(trained.val_indices.numel()))]
    if int(eval_indices.numel()) == 0:
        eval_indices = trained.train_indices[: min(int(cfg.geometry_eval_samples), int(trained.train_indices.numel()))]
    z_eval = encode_weights(vae, trained.normalizer, weights_device.index_select(0, eval_indices.to(device=device)))
    rows: list[dict[str, float | str]] = []
    specs = [
        ("decoder_latent", None, z_eval),
        ("decoder_trained_nf", trained_flow, trained_flow(z_eval)[0].detach()),
        ("decoder_random_nf", random_flow, random_flow(z_eval)[0].detach()),
    ]
    progress = make_progress(cfg, total=len(specs) * int(z_eval.shape[0]), desc="geometry jacobians")
    try:
        for coordinate, flow, samples in specs:
            jacobian_chunks: list[torch.Tensor] = []
            for sample_idx in range(int(samples.shape[0])):
                progress.set_postfix({"coord": coordinate, "sample": f"{sample_idx + 1}/{int(samples.shape[0])}", "latent_dim": int(samples.shape[1])})
                jacobian_chunks.append(decoder_jacobians(vae, trained.normalizer, flow, samples[sample_idx : sample_idx + 1], create_graph=False))
                progress.update(1)
            jac = torch.cat(jacobian_chunks, dim=0)
            metrics = geometry_metrics_from_jacobians(jac)
            rows.append({"coordinate": coordinate, **metrics})
            progress.set_postfix({"coord": coordinate, "R": f"{metrics['isometry_objective']:.4g}", "cond50": f"{metrics['condition_median']:.4g}"})
    finally:
        progress.close()
    return pd.DataFrame(rows)


def _build_interpretation(vae_metrics: pd.DataFrame, geometry: pd.DataFrame, results: pd.DataFrame) -> str:
    quality = vae_metrics[vae_metrics.get("record_type", "") == "vae_quality"] if "record_type" in vae_metrics.columns else pd.DataFrame()
    lines = [
        "## Automatic Interpretation",
        "",
    ]
    variants = ["baseline"]
    if "vae_variant" in vae_metrics.columns:
        variants = [str(v) for v in vae_metrics["vae_variant"].dropna().unique().tolist()]
    for variant in variants:
        qv = quality[quality.get("vae_variant", variant) == variant] if "vae_variant" in quality.columns else quality
        gv = geometry[geometry.get("vae_variant", variant) == variant] if "vae_variant" in geometry.columns else geometry
        rv = results[results.get("vae_variant", variant) == variant] if "vae_variant" in results.columns else results
        decoded_rel = float(qv["reconstruction_rel_l2"].median()) if not qv.empty else float("nan")
        geom = gv.set_index("coordinate") if not gv.empty else pd.DataFrame()
        latent_r = float(geom.loc["decoder_latent", "isometry_objective"]) if "decoder_latent" in geom.index else float("nan")
        trained_r = float(geom.loc["decoder_trained_nf", "isometry_objective"]) if "decoder_trained_nf" in geom.index else float("nan")
        grouped = rv.groupby("method") if not rv.empty else {}

        def median_metric(method: str, column: str) -> float:
            if method not in grouped.groups:
                return float("nan")
            return float(grouped.get_group(method)[column].median())

        latent_aulc = median_metric("decoder_latent", "aulc")
        trained_aulc = median_metric("decoder_trained_nf", "aulc")
        random_aulc = median_metric("decoder_random_nf", "aulc")
        raw_aulc = median_metric("raw", "aulc")
        claim_help = trained_aulc < latent_aulc and trained_aulc < random_aulc
        lines.extend(
            [
                f"### `{variant}`",
                "",
                f"1. VAE reconstruction median relative L2: `{decoded_rel:.4g}`.",
                f"2. Decoder geometry isometry objective: latent `{latent_r:.4g}`, trained NF `{trained_r:.4g}`.",
                f"3. Median AULC: raw `{raw_aulc:.4g}`, decoder latent `{latent_aulc:.4g}`, trained NF `{trained_aulc:.4g}`, random NF `{random_aulc:.4g}`.",
                f"4. Post-hoc smoothing claim allowed: `{bool(claim_help)}`.",
                "5. Trained NF is considered helpful only if it beats both decoder latent and random near-identity NF under selected per-method LRs.",
                "6. If decoded starts have high reconstruction mismatch, interpret downstream failures as VAE-quality limited before blaming smoothing.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_or_load(cfg: ExperimentConfig) -> ExperimentTables:
    output_dir = run_dir(cfg)
    cfg_path = output_dir / "config.json"
    cached_cfg = read_config(cfg_path)
    if bool(cfg.cache_first) and not bool(cfg.force_rerun) and cached_cfg and cached_cfg.get("config_hash") == config_hash(cfg) and _all_outputs_exist(output_dir, cfg):
        return _read_existing(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = write_config(cfg_path, cfg)
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    variants = _vae_variants_from_coeff(float(cfg.vae_geometry_reg_coeff))
    multi_variant = len(variants) > 1
    stage_total = 2 + 5 * len(variants) + 1
    stage_idx = 1
    stage_progress = make_progress(cfg, total=stage_total, desc="sage smoothing pipeline", leave=True)

    try:
        _set_stage(stage_progress, stage_idx, stage_total, "load data")
        train_images, train_labels, test_images, test_labels = load_vision_tensors(cfg)
        train_images = train_images.to(device=device, dtype=dtype)
        train_labels = train_labels.to(device=device)
        test_images = test_images.to(device=device, dtype=dtype)
        test_labels = test_labels.to(device=device)
        stage_progress.update(1)
        stage_idx += 1

        _set_stage(stage_progress, stage_idx, stage_total, "generate weights")
        weights, weight_records, spec = generate_weight_pool(
            cfg,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            output_path=output_dir / "weight_pool.pt",
        )
        weight_records.to_csv(output_dir / "weight_pool_records.csv", index=False)
        weight_cache_key = weight_pool_cache_key(cfg)
        stage_progress.update(1)
        stage_idx += 1

        weights_device = weights.to(device=device, dtype=dtype)
        vae_metric_frames: list[pd.DataFrame] = []
        geometry_frames: list[pd.DataFrame] = []
        downstream_result_frames: list[pd.DataFrame] = []
        downstream_curve_frames: list[pd.DataFrame] = []
        selected_lr_frames: list[pd.DataFrame] = []
        flow_history_frames: list[pd.DataFrame] = []

        for variant, reg_coeff in variants:
            variant_cfg = replace(cfg, vae_geometry_reg_coeff=float(reg_coeff))

            _set_stage(stage_progress, stage_idx, stage_total, f"{variant}: train VAE")
            trained = train_weight_vae(
                variant_cfg,
                weights,
                output_path=_variant_path(output_dir, "vae_checkpoint.pt", variant=variant, multi_variant=multi_variant),
                upstream_cache_key=weight_cache_key,
            )
            trained.vae.to(device=device, dtype=dtype).eval()
            z_all = encode_weights(trained.vae, trained.normalizer, weights_device)
            z_train = z_all.index_select(0, trained.train_indices.to(device=device))
            vae_cache_key = vae_training_cache_key(variant_cfg, weights, upstream_cache_key=weight_cache_key)
            stage_progress.update(1)
            stage_idx += 1

            _set_stage(stage_progress, stage_idx, stage_total, f"{variant}: train NF")
            trained_flow, flow_history = train_posthoc_flow(
                variant_cfg,
                vae=trained.vae,
                normalizer=trained.normalizer,
                z_train=z_train,
                output_path=_variant_path(output_dir, "flow_state.pt", variant=variant, multi_variant=multi_variant),
                upstream_cache_key=vae_cache_key,
            )
            flow_cache_key = flow_training_cache_key(variant_cfg, z_train, upstream_cache_key=vae_cache_key)
            flow_history = flow_history.copy()
            flow_history["vae_variant"] = variant
            flow_history["vae_geometry_reg_coeff"] = float(reg_coeff)
            flow_history_frames.append(flow_history)
            random_flow = random_near_identity_flow(variant_cfg, int(variant_cfg.latent_dim), device=device, dtype=dtype)
            atomic_torch_save(
                {"cache_key": f"random-{flow_cache_key}", "flow_state": random_flow.state_dict()},
                _variant_path(output_dir, "random_flow_state.pt", variant=variant, multi_variant=multi_variant),
            )
            stage_progress.update(1)
            stage_idx += 1

            _set_stage(stage_progress, stage_idx, stage_total, f"{variant}: VAE quality")
            vae_metrics_variant = _vae_quality_rows(
                cfg=variant_cfg,
                trained=trained,
                weights=weights,
                spec=spec,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
            )
            vae_metrics_variant["vae_variant"] = variant
            vae_metrics_variant["vae_geometry_reg_coeff"] = float(reg_coeff)
            vae_metric_frames.append(vae_metrics_variant)
            stage_progress.update(1)
            stage_idx += 1

            _set_stage(stage_progress, stage_idx, stage_total, f"{variant}: geometry")
            geometry_variant = _geometry_table(
                cfg=variant_cfg,
                trained=trained,
                weights=weights,
                trained_flow=trained_flow,
                random_flow=random_flow,
            )
            geometry_variant["vae_variant"] = variant
            geometry_variant["vae_geometry_reg_coeff"] = float(reg_coeff)
            geometry_frames.append(geometry_variant)
            stage_progress.update(1)
            stage_idx += 1

            _set_stage(stage_progress, stage_idx, stage_total, f"{variant}: downstream")
            start_indices = trained.val_indices
            if int(start_indices.numel()) < int(variant_cfg.tune_starts) + int(variant_cfg.eval_starts):
                start_indices = torch.arange(int(weights.shape[0]))
            starts = weights_device.index_select(0, start_indices.to(device=device))
            ctx = DownstreamContext(
                cfg=variant_cfg,
                spec=spec,
                vae=trained.vae,
                normalizer=trained.normalizer,
                trained_flow=trained_flow,
                random_flow=random_flow,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
            )
            downstream_results_variant, downstream_curves_variant, selected_lrs_variant = tune_and_evaluate_downstream(
                cfg=variant_cfg,
                ctx=ctx,
                starts=starts,
            )
            for frame in (downstream_results_variant, downstream_curves_variant, selected_lrs_variant):
                frame["vae_variant"] = variant
                frame["vae_geometry_reg_coeff"] = float(reg_coeff)
            downstream_result_frames.append(downstream_results_variant)
            downstream_curve_frames.append(downstream_curves_variant)
            selected_lr_frames.append(selected_lrs_variant)
            stage_progress.update(1)
            stage_idx += 1

        vae_metrics = pd.concat(vae_metric_frames, ignore_index=True, sort=False)
        geometry = pd.concat(geometry_frames, ignore_index=True, sort=False)
        downstream_results = pd.concat(downstream_result_frames, ignore_index=True, sort=False)
        downstream_curves = pd.concat(downstream_curve_frames, ignore_index=True, sort=False)
        selected_lrs = pd.concat(selected_lr_frames, ignore_index=True, sort=False)
        flow_history_all = pd.concat(flow_history_frames, ignore_index=True, sort=False) if flow_history_frames else pd.DataFrame()
        interpretation = _build_interpretation(vae_metrics, geometry, downstream_results)

        _set_stage(stage_progress, stage_idx, stage_total, "write outputs")
        vae_metrics.to_csv(output_dir / "vae_metrics.csv", index=False)
        geometry.to_csv(output_dir / "geometry.csv", index=False)
        selected_lrs.to_csv(output_dir / "selected_lrs.csv", index=False)
        downstream_results.to_csv(output_dir / "downstream_results.csv", index=False)
        downstream_curves.to_csv(output_dir / "downstream_curves.csv", index=False)
        flow_history_all.to_csv(output_dir / "flow_history.csv", index=False)
        (output_dir / "interpretation.md").write_text(interpretation, encoding="utf-8")
        stage_progress.update(1)
    finally:
        stage_progress.close()
    return ExperimentTables(
        vae_metrics=vae_metrics,
        geometry=geometry,
        selected_lrs=selected_lrs,
        downstream_results=downstream_results,
        downstream_curves=downstream_curves,
        output_dir=output_dir,
        config=config_payload,
        interpretation_markdown=interpretation,
    )
