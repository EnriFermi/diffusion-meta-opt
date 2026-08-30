from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    _stable_uint63,
    load_celo_meta_task_tensors,
    load_torch_cache,
    move_task_tensors,
    spec_from_payload,
    vae_block_direction_loss,
    vae_block_reconstruction_loss,
    vae_function_anchor_loss,
    vae_loss,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    PreconditioningState,
    preconditioning_regularizer,
)
from scripts.analyze_variant_a_margin_mechanism import _load_run, _spec_slices


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_m2048_cap0p25_clip20_fc2recon_lam0p03_marginhuber_c0p001_rank_repair_v1_seed0"
)
DEFAULT_START_BANK = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m2048_h4096/trajectory_discriminator/selected_16_start_bank.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m2048_h4096/objective_conflict_audit"
)


def _log(message: str) -> None:
    print(f"[objective_conflict] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_scalar_metrics(prefix: str, row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, (float, int)):
            scalar = float(value)
            if math.isfinite(scalar):
                metrics[f"{prefix}{key}"] = scalar
    return metrics


def _load_start_bank(path: Path, *, samples: int) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "source_weight_index" not in rows.columns:
        raise ValueError(f"{path} missing source_weight_index")
    rows = rows.drop_duplicates(subset=["source_weight_index"], keep="first").reset_index(drop=True)
    if int(samples) > 0:
        rows = rows.iloc[: int(samples)].copy()
    rows["source_weight_index"] = pd.to_numeric(rows["source_weight_index"], errors="raise").astype(int)
    rows["audit_order"] = np.arange(len(rows), dtype=int)
    return rows


def _named_trainable_params(vae: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in vae.named_parameters() if param.requires_grad]


PARAM_GROUPS = (
    "all",
    "encoder",
    "decoder",
    "latent_to_context",
    "decoder_attention",
    "decoder_ffn",
    "patch_decoder",
    "decoder_scale_head",
)


def _param_groups_for_name(name: str) -> tuple[str, ...]:
    groups = ["all"]
    if name.startswith(("patch_encoder", "resampler_", "head_", "token_norm", "to_mu", "to_logvar")):
        groups.append("encoder")
    if name.startswith(("latent_to_context", "decoder_", "patch_decoder", "decoder_scale_head")):
        groups.append("decoder")
    if name.startswith("latent_to_context"):
        groups.append("latent_to_context")
    if name.startswith(("decoder_cross_attn", "decoder_qkv", "decoder_out_proj", "decoder_attn_norm")):
        groups.append("decoder_attention")
    if name.startswith("decoder_ffn"):
        groups.append("decoder_ffn")
    if name.startswith("patch_decoder"):
        groups.append("patch_decoder")
    if name.startswith("decoder_scale_head"):
        groups.append("decoder_scale_head")
    return tuple(groups)


def _zero_grad(vae: torch.nn.Module) -> None:
    for param in vae.parameters():
        param.grad = None


def _loss_grad_groups(
    *,
    loss: torch.Tensor,
    vae: torch.nn.Module,
    named_params: list[tuple[str, torch.nn.Parameter]],
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    _zero_grad(vae)
    loss.backward(retain_graph=retain_graph)
    group_values: dict[str, list[torch.Tensor]] = {group: [] for group in PARAM_GROUPS}
    for name, param in named_params:
        grad = param.grad.detach() if param.grad is not None else torch.zeros_like(param.detach())
        flat = grad.reshape(-1)
        for group in _param_groups_for_name(name):
            group_values[group].append(flat)
    out: dict[str, torch.Tensor] = {}
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    for group, values in group_values.items():
        if values:
            out[group] = torch.cat([v.reshape(-1) for v in values]).detach()
        else:
            out[group] = torch.zeros(0, device=device, dtype=dtype)
    _zero_grad(vae)
    return out


def _flat_block_groups(vector: torch.Tensor, *, slices: dict[str, slice]) -> dict[str, torch.Tensor]:
    flat = vector.reshape(-1) if vector.ndim == 1 else vector.reshape(-1, int(vector.shape[-1]))
    groups: dict[str, torch.Tensor] = {"all": flat.reshape(-1)}
    fc2_weight_slice = slices.get("fc2.weight")
    fc2_bias_slice = slices.get("fc2.bias")
    if fc2_weight_slice is not None:
        groups["fc2.weight"] = flat[..., fc2_weight_slice].reshape(-1)
    if fc2_bias_slice is not None:
        groups["fc2.bias"] = flat[..., fc2_bias_slice].reshape(-1)
    classifier_parts = [groups[key] for key in ("fc2.weight", "fc2.bias") if key in groups]
    if classifier_parts:
        groups["classifier_head"] = torch.cat([part.reshape(-1) for part in classifier_parts], dim=0)
    return groups


def _decoded_loss_grad_groups(
    *,
    loss: torch.Tensor,
    recon_norm: torch.Tensor,
    normalizer: Any,
    slices: dict[str, slice],
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    grad_norm = torch.autograd.grad(loss, recon_norm, retain_graph=retain_graph, allow_unused=True)[0]
    if grad_norm is None:
        grad_norm = torch.zeros_like(recon_norm)
    std = normalizer.std.to(device=grad_norm.device, dtype=grad_norm.dtype).clamp_min(float(normalizer.eps))
    grad_raw = grad_norm / std.reshape(1, -1)
    out: dict[str, torch.Tensor] = {}
    for block, values in _flat_block_groups(grad_norm.detach(), slices=slices).items():
        out[f"norm:{block}"] = values
    for block, values in _flat_block_groups(grad_raw.detach(), slices=slices).items():
        out[f"raw:{block}"] = values
    return out


def _dot_cos(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float, float]:
    if int(a.numel()) == 0 or int(b.numel()) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = a.reshape(-1).to(dtype=torch.float64)
    b = b.reshape(-1).to(dtype=torch.float64)
    dot = float(torch.dot(a, b).detach().cpu().item())
    an = float(a.norm().detach().cpu().item())
    bn = float(b.norm().detach().cpu().item())
    cos = dot / max(an * bn, 1e-300)
    return dot, an, bn, cos


def _chunked_sources(start_bank: pd.DataFrame, *, chunk_size: int) -> list[list[int]]:
    sources = [int(v) for v in start_bank["source_weight_index"].tolist()]
    size = max(1, int(chunk_size))
    return [sources[i : i + size] for i in range(0, len(sources), size)]


def _records_for_sources(records: pd.DataFrame, sources: list[int]) -> list[dict[str, Any]]:
    out = records.iloc[[int(v) for v in sources]].to_dict(orient="records")
    for record, source in zip(out, sources, strict=True):
        record["source_weight_index"] = int(source)
    return out


def _compute_losses_for_chunk(
    *,
    run: dict[str, Any],
    sources: list[int],
    step: int,
    precond_state: PreconditioningState,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    cfg = run["cfg"]
    vae = run["vae"]
    normalizer = run["normalizer"]
    spec = run["spec"]
    weights = run["weights"]
    records = run["records"]
    task_tensors = run["task_tensors"]
    device = run["device"]
    batch_indices = torch.tensor([int(v) for v in sources], device=device, dtype=torch.long)
    batch_raw = weights.index_select(0, batch_indices).detach()
    batch = normalizer.normalize(batch_raw)
    vae.eval()
    recon, mu, logvar = vae(batch)
    base_loss, base_row = vae_loss(
        cfg,
        batch,
        batch_raw,
        recon,
        mu,
        logvar,
        normalizer=normalizer,
        spec=spec,
    )
    losses: dict[str, torch.Tensor] = {"base": base_loss}
    metrics: dict[str, float] = _finite_scalar_metrics("base_", base_row)
    block_coeff = max(0.0, float(getattr(cfg, "vae_block_recon_coeff", 0.0)))
    if block_coeff > 0.0:
        block_recon = vae.decode_norm(mu)
        block_loss, block_row = vae_block_reconstruction_loss(
            cfg,
            x_norm=batch,
            target_weights=batch_raw,
            recon_norm=block_recon,
            normalizer=normalizer,
            spec=spec,
            step=int(step),
        )
        block_effective_coeff = float(block_row["block_recon_coeff"]) * float(block_row["block_recon_ramp"])
        losses["block_recon"] = block_loss * float(block_effective_coeff)
        metrics.update(_finite_scalar_metrics("block_", block_row))
    direction_coeff = max(0.0, float(getattr(cfg, "vae_block_direction_coeff", 0.0)))
    if direction_coeff > 0.0:
        direction_recon = vae.decode_norm(mu)
        direction_loss, direction_row = vae_block_direction_loss(
            cfg,
            x_norm=batch,
            target_weights=batch_raw,
            recon_norm=direction_recon,
            normalizer=normalizer,
            spec=spec,
            step=int(step),
        )
        direction_effective_coeff = float(direction_row["block_direction_coeff"]) * float(direction_row["block_direction_ramp"])
        losses["block_direction"] = direction_loss * float(direction_effective_coeff)
        metrics.update(_finite_scalar_metrics("direction_", direction_row))
    anchor_coeff = max(0.0, float(getattr(cfg, "vae_function_anchor_coeff", 0.0)))
    if anchor_coeff > 0.0:
        anchor_recon = vae.decode_norm(mu)
        anchor_loss, anchor_row = vae_function_anchor_loss(
            cfg,
            recon_norm=anchor_recon,
            batch_raw=batch_raw,
            batch_indices=batch_indices,
            normalizer=normalizer,
            weight_records=records,
            task_tensors=task_tensors,
            spec=spec,
            step=int(step),
        )
        losses["function_anchor"] = anchor_loss * float(anchor_coeff)
        metrics.update(_finite_scalar_metrics("anchor_", anchor_row))
    precond_kind = str(getattr(cfg, "vae_precond_loss_kind", "none")).strip().lower()
    if precond_kind not in {"", "none", "off", "disabled"}:
        precond_count = min(max(1, int(cfg.vae_precond_samples)), int(mu.shape[0]))
        z_precond = mu[:precond_count].detach()
        precond_records = _records_for_sources(records, sources[:precond_count])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_uint63("precond_probe", int(cfg.seed), int(step)))
        precond_loss, precond_row = preconditioning_regularizer(
            cfg,
            state=precond_state,
            vae=vae,
            normalizer=normalizer,
            z_samples=z_precond,
            records=precond_records,
            task_tensors=task_tensors,
            spec=spec,
            step=int(step),
            generator=generator,
        )
        ramp_steps = max(0, int(getattr(cfg, "vae_precond_ramp_steps", 0)))
        warmup = max(0, int(getattr(cfg, "vae_precond_warmup_steps", 0)))
        precond_scale = 1.0 if ramp_steps <= 0 else min(1.0, max(0.0, float(int(step) - warmup) / float(ramp_steps)))
        losses["precond"] = precond_loss * float(precond_scale)
        metrics.update(_finite_scalar_metrics("precond_", precond_row))
        metrics["precond_ramp_scale"] = float(precond_scale)
    guard_terms = [losses[name] for name in ["block_recon", "block_direction", "function_anchor"] if name in losses]
    losses["guards"] = sum(guard_terms, start=base_loss.new_zeros(()))
    losses["base_plus_guards"] = losses["base"] + losses["guards"]
    return losses, metrics


def _compute_decoded_losses_for_chunk(
    *,
    run: dict[str, Any],
    sources: list[int],
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    cfg = run["cfg"]
    vae = run["vae"]
    normalizer = run["normalizer"]
    spec = run["spec"]
    weights = run["weights"]
    records = run["records"]
    task_tensors = run["task_tensors"]
    device = run["device"]
    batch_indices = torch.tensor([int(v) for v in sources], device=device, dtype=torch.long)
    batch_raw = weights.index_select(0, batch_indices).detach()
    batch = normalizer.normalize(batch_raw)
    vae.eval()
    with torch.no_grad():
        recon0, mu0, logvar0 = vae(batch)
    recon = recon0.detach().requires_grad_(True)
    base_loss, base_row = vae_loss(
        cfg,
        batch,
        batch_raw,
        recon,
        mu0.detach(),
        logvar0.detach(),
        normalizer=normalizer,
        spec=spec,
    )
    losses: dict[str, torch.Tensor] = {"base": base_loss}
    metrics: dict[str, float] = _finite_scalar_metrics("decoded_base_", base_row)
    block_coeff = max(0.0, float(getattr(cfg, "vae_block_recon_coeff", 0.0)))
    if block_coeff > 0.0:
        block_loss, block_row = vae_block_reconstruction_loss(
            cfg,
            x_norm=batch,
            target_weights=batch_raw,
            recon_norm=recon,
            normalizer=normalizer,
            spec=spec,
            step=int(step),
        )
        block_effective_coeff = float(block_row["block_recon_coeff"]) * float(block_row["block_recon_ramp"])
        losses["block_recon"] = block_loss * float(block_effective_coeff)
        metrics.update(_finite_scalar_metrics("decoded_block_", block_row))
    direction_coeff = max(0.0, float(getattr(cfg, "vae_block_direction_coeff", 0.0)))
    if direction_coeff > 0.0:
        direction_loss, direction_row = vae_block_direction_loss(
            cfg,
            x_norm=batch,
            target_weights=batch_raw,
            recon_norm=recon,
            normalizer=normalizer,
            spec=spec,
            step=int(step),
        )
        direction_effective_coeff = float(direction_row["block_direction_coeff"]) * float(direction_row["block_direction_ramp"])
        losses["block_direction"] = direction_loss * float(direction_effective_coeff)
        metrics.update(_finite_scalar_metrics("decoded_direction_", direction_row))
    anchor_coeff = max(0.0, float(getattr(cfg, "vae_function_anchor_coeff", 0.0)))
    if anchor_coeff > 0.0:
        anchor_loss, anchor_row = vae_function_anchor_loss(
            cfg,
            recon_norm=recon,
            batch_raw=batch_raw,
            batch_indices=batch_indices,
            normalizer=normalizer,
            weight_records=records,
            task_tensors=task_tensors,
            spec=spec,
            step=int(step),
        )
        losses["function_anchor"] = anchor_loss * float(anchor_coeff)
        metrics.update(_finite_scalar_metrics("decoded_anchor_", anchor_row))
    guard_terms = [losses[name] for name in ["block_recon", "block_direction", "function_anchor"] if name in losses]
    losses["guards"] = sum(guard_terms, start=base_loss.new_zeros(()))
    losses["base_plus_guards"] = losses["base"] + losses["guards"]
    return losses, metrics, recon


def _summarize_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (pair, group_name), group in rows.groupby(["pair", "group"], sort=True):
        cos = pd.to_numeric(group["cosine"], errors="coerce")
        finite = cos[np.isfinite(cos.to_numpy(dtype=float))]
        summaries.append(
            {
                "pair": pair,
                "group": group_name,
                "chunks": int(len(group)),
                "finite_chunks": int(len(finite)),
                "cosine_mean": float(finite.mean()) if len(finite) else float("nan"),
                "cosine_median": float(finite.median()) if len(finite) else float("nan"),
                "cosine_min": float(finite.min()) if len(finite) else float("nan"),
                "cosine_max": float(finite.max()) if len(finite) else float("nan"),
                "negative_cosine_fraction": float((finite < 0.0).mean()) if len(finite) else float("nan"),
            }
        )
    return pd.DataFrame(summaries)


def _summarize_terms(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (space, term, group_name), group in rows.groupby(["space", "term", "group"], sort=True):
        norms = pd.to_numeric(group["grad_norm"], errors="coerce")
        finite = norms[np.isfinite(norms.to_numpy(dtype=float))]
        summaries.append(
            {
                "space": space,
                "term": term,
                "group": group_name,
                "chunks": int(len(group)),
                "finite_chunks": int(len(finite)),
                "grad_norm_mean": float(finite.mean()) if len(finite) else float("nan"),
                "grad_norm_median": float(finite.median()) if len(finite) else float("nan"),
                "grad_norm_max": float(finite.max()) if len(finite) else float("nan"),
            }
        )
    return pd.DataFrame(summaries)


def _numeric_table_finite(table: pd.DataFrame, *, optional_substrings: tuple[str, ...] = ()) -> tuple[bool, dict[str, int]]:
    bad: dict[str, int] = {}
    numeric = table.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if any(fragment in str(col) for fragment in optional_substrings):
            continue
        values = pd.to_numeric(numeric[col], errors="coerce").to_numpy(dtype=np.float64)
        bad_count = int((~np.isfinite(values)).sum())
        if bad_count:
            bad[str(col)] = bad_count
    return (len(bad) == 0), bad


def _write_review(
    out_dir: Path,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    decoded_summary: pd.DataFrame,
    term_summary: pd.DataFrame,
) -> None:
    focus = summary[
        summary["pair"].astype(str).isin(
            ["precond_vs_function_anchor", "precond_vs_block_recon", "precond_vs_base_plus_guards"]
        )
        & summary["group"].astype(str).isin(["all", "decoder", "latent_to_context", "patch_decoder"])
    ].copy()
    decoded_focus = decoded_summary[
        decoded_summary["pair"].astype(str).isin(["block_recon_vs_function_anchor", "base_vs_function_anchor"])
        & decoded_summary["group"].astype(str).isin(["norm:fc2.weight", "norm:classifier_head", "raw:fc2.weight", "raw:classifier_head"])
    ].copy()
    term_focus = term_summary[
        term_summary["space"].astype(str).eq("decoded")
        & term_summary["term"].astype(str).isin(["block_recon", "function_anchor"])
        & term_summary["group"].astype(str).isin(["norm:fc2.weight", "norm:classifier_head", "raw:fc2.weight", "raw:classifier_head"])
    ].copy()
    lines = [
        "# Objective-Conflict Gradient Audit Review",
        "",
        "## Validity",
        f"- Accepted: `{validation.get('accepted')}`.",
        f"- Run: `{validation.get('run')}`.",
        f"- Starts: `{validation.get('starts')}`; chunks: `{validation.get('chunks')}`; step: `{validation.get('step')}`.",
        f"- Required finite numeric metrics: `{validation.get('all_required_numeric_finite')}`.",
        "",
        "## VAE-Param Conflict Focus",
        focus.to_markdown(index=False) if not focus.empty else "_missing_",
        "",
        "## Decoded-Space Guard Focus",
        decoded_focus.to_markdown(index=False) if not decoded_focus.empty else "_missing_",
        "",
        "## Decoded-Space Term Norms",
        term_focus.to_markdown(index=False) if not term_focus.empty else "_missing_",
        "",
        "## Interpretation Boundary",
        "Negative VAE-param cosine means the gradient descent step for the preconditioning loss would increase the other objective locally. The decoded-space rows deliberately omit precond because the A surrogate depends on decoder Jacobian/HVP structure, not only on a single decoded weight value.",
    ]
    (out_dir / "objective_conflict_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Posthoc gradient conflict audit for Variant A objective terms.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--start-bank-csv", type=Path, default=DEFAULT_START_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--step", type=int, default=-1)
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "startup "
        f"device={args.device} run={args.run} start_bank={args.start_bank_csv} "
        f"samples={args.samples} chunk_size={args.chunk_size} output_dir={out_dir}"
    )
    start_bank = _load_start_bank(args.start_bank_csv, samples=int(args.samples))
    run = _load_run(str(args.run), device=str(args.device))
    step = int(args.step) if int(args.step) > 0 else int(run["cfg"].vae_steps)
    slices = _spec_slices(run["spec"])
    chunks = _chunked_sources(start_bank, chunk_size=int(args.chunk_size))
    named_params = _named_trainable_params(run["vae"])
    precond_state = PreconditioningState()
    pair_specs = [
        ("precond_vs_base", "precond", "base"),
        ("precond_vs_block_recon", "precond", "block_recon"),
        ("precond_vs_function_anchor", "precond", "function_anchor"),
        ("precond_vs_guards", "precond", "guards"),
        ("precond_vs_base_plus_guards", "precond", "base_plus_guards"),
        ("block_recon_vs_function_anchor", "block_recon", "function_anchor"),
    ]
    pair_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    decoded_pair_rows: list[dict[str, Any]] = []
    decoded_term_rows: list[dict[str, Any]] = []
    decoded_pair_specs = [
        ("base_vs_function_anchor", "base", "function_anchor"),
        ("block_recon_vs_function_anchor", "block_recon", "function_anchor"),
        ("base_plus_guards_vs_function_anchor", "base_plus_guards", "function_anchor"),
        ("base_vs_block_recon", "base", "block_recon"),
    ]
    for chunk_id, sources in enumerate(chunks):
        _log(f"stage=chunk {chunk_id + 1}/{len(chunks)} sources={sources}")
        losses, metrics = _compute_losses_for_chunk(run=run, sources=sources, step=step, precond_state=precond_state)
        grad_by_term: dict[str, dict[str, torch.Tensor]] = {}
        term_names = list(losses.keys())
        for idx, term in enumerate(term_names):
            if not bool(losses[term].requires_grad):
                continue
            grad_by_term[term] = _loss_grad_groups(
                loss=losses[term],
                vae=run["vae"],
                named_params=named_params,
                retain_graph=idx < len(term_names) - 1,
            )
            for group_name, vector in grad_by_term[term].items():
                term_rows.append(
                    {
                        "space": "vae_param",
                        "chunk_id": int(chunk_id),
                        "sources": ",".join(str(v) for v in sources),
                        "term": term,
                        "group": group_name,
                        "loss_value": float(losses[term].detach().cpu().item()),
                        "grad_norm": float(vector.to(dtype=torch.float64).norm().detach().cpu().item()) if int(vector.numel()) else 0.0,
                        **metrics,
                    }
                )
        for pair_name, left, right in pair_specs:
            if left not in grad_by_term or right not in grad_by_term:
                continue
            for group_name in sorted(set(grad_by_term[left]) & set(grad_by_term[right])):
                dot, left_norm, right_norm, cos = _dot_cos(grad_by_term[left][group_name], grad_by_term[right][group_name])
                pair_rows.append(
                    {
                        "chunk_id": int(chunk_id),
                        "sources": ",".join(str(v) for v in sources),
                        "pair": pair_name,
                        "left_term": left,
                        "right_term": right,
                        "group": group_name,
                        "dot": dot,
                        "left_grad_norm": left_norm,
                        "right_grad_norm": right_norm,
                        "cosine": cos,
                        "precond_descent_increases_right_loss": bool(cos < 0.0) if math.isfinite(cos) and left == "precond" else False,
                    }
                )
        decoded_losses, decoded_metrics, recon_leaf = _compute_decoded_losses_for_chunk(run=run, sources=sources, step=step)
        decoded_grad_by_term: dict[str, dict[str, torch.Tensor]] = {}
        decoded_term_names = list(decoded_losses.keys())
        for idx, term in enumerate(decoded_term_names):
            if not bool(decoded_losses[term].requires_grad):
                continue
            decoded_grad_by_term[term] = _decoded_loss_grad_groups(
                loss=decoded_losses[term],
                recon_norm=recon_leaf,
                normalizer=run["normalizer"],
                slices=slices,
                retain_graph=idx < len(decoded_term_names) - 1,
            )
            for group_name, vector in decoded_grad_by_term[term].items():
                decoded_term_rows.append(
                    {
                        "space": "decoded",
                        "chunk_id": int(chunk_id),
                        "sources": ",".join(str(v) for v in sources),
                        "term": term,
                        "group": group_name,
                        "loss_value": float(decoded_losses[term].detach().cpu().item()),
                        "grad_norm": float(vector.to(dtype=torch.float64).norm().detach().cpu().item()) if int(vector.numel()) else 0.0,
                        **decoded_metrics,
                    }
                )
        for pair_name, left, right in decoded_pair_specs:
            if left not in decoded_grad_by_term or right not in decoded_grad_by_term:
                continue
            for group_name in sorted(set(decoded_grad_by_term[left]) & set(decoded_grad_by_term[right])):
                dot, left_norm, right_norm, cos = _dot_cos(decoded_grad_by_term[left][group_name], decoded_grad_by_term[right][group_name])
                decoded_pair_rows.append(
                    {
                        "chunk_id": int(chunk_id),
                        "sources": ",".join(str(v) for v in sources),
                        "pair": pair_name,
                        "left_term": left,
                        "right_term": right,
                        "group": group_name,
                        "dot": dot,
                        "left_grad_norm": left_norm,
                        "right_grad_norm": right_norm,
                        "cosine": cos,
                    }
                )
    pair_df = pd.DataFrame(pair_rows)
    term_df = pd.DataFrame(term_rows)
    decoded_pair_df = pd.DataFrame(decoded_pair_rows)
    decoded_term_df = pd.DataFrame(decoded_term_rows)
    summary = _summarize_pairs(pair_df)
    decoded_summary = _summarize_pairs(decoded_pair_df)
    term_summary = _summarize_terms(pd.concat([term_df, decoded_term_df], axis=0, ignore_index=True))
    pair_df.to_csv(out_dir / "objective_conflict_grad_rows.csv", index=False)
    term_df.to_csv(out_dir / "objective_conflict_term_rows.csv", index=False)
    summary.to_csv(out_dir / "objective_conflict_grad_summary.csv", index=False)
    decoded_pair_df.to_csv(out_dir / "objective_conflict_decoded_grad_rows.csv", index=False)
    decoded_term_df.to_csv(out_dir / "objective_conflict_decoded_term_rows.csv", index=False)
    decoded_summary.to_csv(out_dir / "objective_conflict_decoded_grad_summary.csv", index=False)
    term_summary.to_csv(out_dir / "objective_conflict_term_summary.csv", index=False)
    finite_tables: dict[str, bool] = {}
    nonfinite_counts: dict[str, dict[str, int]] = {}
    for name, table in [
        ("objective_conflict_grad_rows", pair_df),
        ("objective_conflict_term_rows", term_df),
        ("objective_conflict_grad_summary", summary),
        ("objective_conflict_decoded_grad_rows", decoded_pair_df),
        ("objective_conflict_decoded_term_rows", decoded_term_df),
        ("objective_conflict_decoded_grad_summary", decoded_summary),
        ("objective_conflict_term_summary", term_summary),
    ]:
        ok, bad = _numeric_table_finite(table)
        finite_tables[name] = bool(ok)
        if bad:
            nonfinite_counts[name] = bad
    all_required_finite = bool(all(finite_tables.values()))
    validation = {
        "accepted": False,
        "run": str(args.run),
        "output_dir": str(out_dir),
        "start_bank_csv": str(args.start_bank_csv),
        "start_bank_sha256": _sha256_file(args.start_bank_csv),
        "starts": int(len(start_bank)),
        "chunks": int(len(chunks)),
        "chunk_size": int(args.chunk_size),
        "step": int(step),
        "pair_rows": int(len(pair_df)),
        "term_rows": int(len(term_df)),
        "summary_rows": int(len(summary)),
        "decoded_pair_rows": int(len(decoded_pair_df)),
        "decoded_term_rows": int(len(decoded_term_df)),
        "decoded_summary_rows": int(len(decoded_summary)),
        "term_summary_rows": int(len(term_summary)),
        "param_groups": list(PARAM_GROUPS),
        "has_precond_pairs": bool((pair_df["left_term"].astype(str) == "precond").any()) if not pair_df.empty else False,
        "has_decoded_function_anchor": bool((decoded_term_df["term"].astype(str) == "function_anchor").any()) if not decoded_term_df.empty else False,
        "finite_tables": finite_tables,
        "nonfinite_numeric_counts": nonfinite_counts,
        "all_required_numeric_finite": all_required_finite,
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    validation["accepted"] = bool(
        validation["starts"] > 0
        and validation["chunks"] > 0
        and validation["pair_rows"] > 0
        and validation["term_rows"] > 0
        and validation["decoded_term_rows"] > 0
        and validation["has_precond_pairs"]
        and validation["has_decoded_function_anchor"]
        and validation["all_required_numeric_finite"]
    )
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "script": "scripts/audit_variant_a_objective_conflict.py",
        "validation": validation,
        "artifacts": [
            "objective_conflict_grad_rows.csv",
            "objective_conflict_term_rows.csv",
            "objective_conflict_grad_summary.csv",
            "objective_conflict_decoded_grad_rows.csv",
            "objective_conflict_decoded_term_rows.csv",
            "objective_conflict_decoded_grad_summary.csv",
            "objective_conflict_term_summary.csv",
            "validation.json",
            "objective_conflict_review.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_review(out_dir, validation, summary, decoded_summary, term_summary)
    _log(f"done accepted={validation['accepted']} elapsed_sec={validation['elapsed_sec']:.1f} review={out_dir / 'objective_conflict_review.md'}")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
