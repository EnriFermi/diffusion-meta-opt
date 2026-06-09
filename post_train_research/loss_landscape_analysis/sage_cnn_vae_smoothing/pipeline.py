from __future__ import annotations

import json
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
)
from .downstream import DownstreamContext, tune_and_evaluate_downstream


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


def _all_outputs_exist(output_dir: Path) -> bool:
    required = (
        "config.json",
        "weight_pool.pt",
        "vae_checkpoint.pt",
        "flow_state.pt",
        "random_flow_state.pt",
        "vae_metrics.csv",
        "geometry.csv",
        "selected_lrs.csv",
        "downstream_results.csv",
        "downstream_curves.csv",
    )
    return all((output_dir / name).is_file() for name in required)


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
    for local_idx, idx in enumerate(eval_indices.tolist()):
        w = weights_device[int(idx)]
        z = encode_weights(vae, trained.normalizer, w.reshape(1, -1)).squeeze(0)
        recon = decode_weights(vae, trained.normalizer, z.reshape(1, -1)).squeeze(0)
        z_roundtrip = encode_weights(vae, trained.normalizer, recon.reshape(1, -1)).squeeze(0)
        raw_train_loss, raw_train_acc = evaluate_flat_model(w, images=train_images, labels=train_labels, spec=spec)
        dec_train_loss, dec_train_acc = evaluate_flat_model(recon, images=train_images, labels=train_labels, spec=spec)
        raw_test_loss, raw_test_acc = evaluate_flat_model(w, images=test_images, labels=test_labels, spec=spec)
        dec_test_loss, dec_test_acc = evaluate_flat_model(recon, images=test_images, labels=test_labels, spec=spec)
        rows.append(
            {
                "sample_index": int(local_idx),
                "weight_index": int(idx),
                "reconstruction_mse": float((recon - w).square().mean().detach().cpu().item()),
                "reconstruction_rel_l2": float(((recon - w).float().norm() / w.float().norm().clamp_min(1e-12)).detach().cpu().item()),
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
    for coordinate, flow, samples in specs:
        jac = decoder_jacobians(vae, trained.normalizer, flow, samples, create_graph=False)
        metrics = geometry_metrics_from_jacobians(jac)
        rows.append({"coordinate": coordinate, **metrics})
    return pd.DataFrame(rows)


def _build_interpretation(vae_metrics: pd.DataFrame, geometry: pd.DataFrame, results: pd.DataFrame) -> str:
    quality = vae_metrics[vae_metrics.get("record_type", "") == "vae_quality"] if "record_type" in vae_metrics.columns else pd.DataFrame()
    decoded_rel = float(quality["reconstruction_rel_l2"].median()) if not quality.empty else float("nan")
    geom = geometry.set_index("coordinate") if not geometry.empty else pd.DataFrame()
    latent_r = float(geom.loc["decoder_latent", "isometry_objective"]) if "decoder_latent" in geom.index else float("nan")
    trained_r = float(geom.loc["decoder_trained_nf", "isometry_objective"]) if "decoder_trained_nf" in geom.index else float("nan")
    grouped = results.groupby("method") if not results.empty else {}

    def median_metric(method: str, column: str) -> float:
        if method not in grouped.groups:
            return float("nan")
        return float(grouped.get_group(method)[column].median())

    latent_aulc = median_metric("decoder_latent", "aulc")
    trained_aulc = median_metric("decoder_trained_nf", "aulc")
    random_aulc = median_metric("decoder_random_nf", "aulc")
    raw_aulc = median_metric("raw", "aulc")
    claim_help = trained_aulc < latent_aulc and trained_aulc < random_aulc
    lines = [
        "## Automatic Interpretation",
        "",
        f"1. VAE reconstruction median relative L2: `{decoded_rel:.4g}`.",
        f"2. Decoder geometry isometry objective: latent `{latent_r:.4g}`, trained NF `{trained_r:.4g}`.",
        f"3. Median AULC: raw `{raw_aulc:.4g}`, decoder latent `{latent_aulc:.4g}`, trained NF `{trained_aulc:.4g}`, random NF `{random_aulc:.4g}`.",
        f"4. Post-hoc smoothing claim allowed: `{bool(claim_help)}`.",
        "5. Trained NF is considered helpful only if it beats both decoder latent and random near-identity NF under selected per-method LRs.",
        "6. If decoded starts have high reconstruction mismatch, interpret downstream failures as VAE-quality limited before blaming smoothing.",
    ]
    return "\n".join(lines) + "\n"


def run_or_load(cfg: ExperimentConfig) -> ExperimentTables:
    output_dir = run_dir(cfg)
    cfg_path = output_dir / "config.json"
    cached_cfg = read_config(cfg_path)
    if bool(cfg.cache_first) and not bool(cfg.force_rerun) and cached_cfg and cached_cfg.get("config_hash") == config_hash(cfg) and _all_outputs_exist(output_dir):
        return _read_existing(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = write_config(cfg_path, cfg)
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)

    train_images, train_labels, test_images, test_labels = load_vision_tensors(cfg)
    train_images = train_images.to(device=device, dtype=dtype)
    train_labels = train_labels.to(device=device)
    test_images = test_images.to(device=device, dtype=dtype)
    test_labels = test_labels.to(device=device)

    weights, weight_records, spec = generate_weight_pool(
        cfg,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        output_path=output_dir / "weight_pool.pt",
    )
    weight_records.to_csv(output_dir / "weight_pool_records.csv", index=False)
    trained = train_weight_vae(cfg, weights, output_path=output_dir / "vae_checkpoint.pt")
    trained.vae.to(device=device, dtype=dtype).eval()
    weights_device = weights.to(device=device, dtype=dtype)
    z_all = encode_weights(trained.vae, trained.normalizer, weights_device)
    z_train = z_all.index_select(0, trained.train_indices.to(device=device))
    trained_flow, flow_history = train_posthoc_flow(
        cfg,
        vae=trained.vae,
        normalizer=trained.normalizer,
        z_train=z_train,
        output_path=output_dir / "flow_state.pt",
    )
    flow_history.to_csv(output_dir / "flow_history.csv", index=False)
    random_flow = random_near_identity_flow(cfg, int(cfg.latent_dim), device=device, dtype=dtype)
    torch.save({"flow_state": random_flow.state_dict()}, output_dir / "random_flow_state.pt")

    vae_metrics = _vae_quality_rows(
        cfg=cfg,
        trained=trained,
        weights=weights,
        spec=spec,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    geometry = _geometry_table(cfg=cfg, trained=trained, weights=weights, trained_flow=trained_flow, random_flow=random_flow)
    start_indices = trained.val_indices
    if int(start_indices.numel()) < int(cfg.tune_starts) + int(cfg.eval_starts):
        start_indices = torch.arange(int(weights.shape[0]))
    starts = weights_device.index_select(0, start_indices.to(device=device))
    ctx = DownstreamContext(
        cfg=cfg,
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
    downstream_results, downstream_curves, selected_lrs = tune_and_evaluate_downstream(cfg=cfg, ctx=ctx, starts=starts)
    interpretation = _build_interpretation(vae_metrics, geometry, downstream_results)

    vae_metrics.to_csv(output_dir / "vae_metrics.csv", index=False)
    geometry.to_csv(output_dir / "geometry.csv", index=False)
    selected_lrs.to_csv(output_dir / "selected_lrs.csv", index=False)
    downstream_results.to_csv(output_dir / "downstream_results.csv", index=False)
    downstream_curves.to_csv(output_dir / "downstream_curves.csv", index=False)
    (output_dir / "interpretation.md").write_text(interpretation, encoding="utf-8")
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
