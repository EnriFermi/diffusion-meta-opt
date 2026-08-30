from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from training.weightclip_benchmark.analyze_polar_tail_mechanism import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    _effective_rank,
    _load_batch,
    _pairwise_cosine_summary,
)
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    LATENT_ROOTED_POLAR_TAIL_SCHEMA,
    LatentRootedPolarTailedUnifiedWeightBottleneck,
    PolarTailedUnifiedWeightBottleneck,
    _load_config,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    PreNormTransformerBlock,
    _seed_everything,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _cross_attention_write(
    block: PreNormTransformerBlock,
    queries: torch.Tensor,
    values: torch.Tensor,
    value_valid: torch.Tensor | None,
) -> torch.Tensor:
    """Use an existing self-attention block as Q <- K,V cross-attention."""
    batch, query_length, dim = queries.shape
    value_length = values.shape[1]
    qkv_weight = block.qkv.weight
    q_weight, k_weight, v_weight = qkv_weight.split(dim, dim=0)
    q = F.linear(block.attn_norm(queries), q_weight).view(
        batch, query_length, block.heads, block.head_dim
    )
    normalized_values = block.attn_norm(values)
    k = F.linear(normalized_values, k_weight).view(
        batch, value_length, block.heads, block.head_dim
    )
    v = F.linear(normalized_values, v_weight).view(
        batch, value_length, block.heads, block.head_dim
    )
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    if block.bounded_cosine_attention:
        qh = F.normalize(qh.float(), dim=-1, eps=1.0e-6).to(q.dtype)
        kh = F.normalize(kh.float(), dim=-1, eps=1.0e-6).to(k.dtype)
    attended = F.scaled_dot_product_attention(
        qh,
        kh,
        v.transpose(1, 2),
        attn_mask=(value_valid[:, None, None, :] if value_valid is not None else None),
        dropout_p=0.0,
        scale=(block.attention_logit_scale if block.bounded_cosine_attention else None),
    )
    return block.attn_out(attended.transpose(1, 2).reshape(batch, query_length, dim))


def _mlp_update(block: PreNormTransformerBlock, state: torch.Tensor) -> torch.Tensor:
    left, right = block.mlp_in(block.mlp_norm(state)).chunk(2, dim=-1)
    hidden = F.silu(left) * right
    return state + block.mlp_out(hidden)


def _rooted_block(
    block: PreNormTransformerBlock,
    queries: torch.Tensor,
    values: torch.Tensor,
    value_valid: torch.Tensor | None,
) -> torch.Tensor:
    # Deliberately no `+ queries`: addresses cannot enter the value stream.
    return _mlp_update(block, _cross_attention_write(block, queries, values, value_valid))


def _decoder_queries(
    model: PolarTailedUnifiedWeightBottleneck,
    z: torch.Tensor,
    tile_row: torch.Tensor,
    tile_col: torch.Tensor,
    token_valid: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    queries = model.output_queries[None].expand(z.shape[0], -1, -1)
    queries = queries + model._tile_position_embedding(tile_row, tile_col)[:, None]
    expanded_context = context.unsqueeze(1).expand(-1, 128, -1, -1).reshape(
        z.shape[0], model.weight_token_count, model.cfg.distribution_d_dist
    )
    queries = model.decoder_query_conditioner(
        torch.cat((queries, expanded_context.to(queries.dtype)), dim=-1)
    )
    return queries * token_valid.unsqueeze(-1).to(queries.dtype)


def _latent_rooted_decode(
    model: PolarTailedUnifiedWeightBottleneck,
    z: torch.Tensor,
    tile_row: torch.Tensor,
    tile_col: torch.Tensor,
    token_valid: torch.Tensor,
    p16_valid: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = model.from_latent(z)
    latent_valid = torch.ones(
        z.shape[0], model.cfg.latent_slots, device=z.device, dtype=torch.bool
    )
    p32_queries = _decoder_queries(
        model, z, tile_row, tile_col, token_valid, context
    )
    state = _rooted_block(
        model.decoder_blocks[0], p32_queries, latent, latent_valid
    )
    state = state * token_valid.unsqueeze(-1).to(state.dtype)
    for block in model.decoder_blocks[1:]:
        state = block(state, valid_mask=token_valid)

    batch, p32_count, dim = state.shape
    if p32_count != model.weight_token_count:
        raise RuntimeError("p32 state count drifted")
    base_child_queries = p32_queries[:, :, None, :].expand(-1, -1, 2, -1)

    def tail(
        block: PreNormTransformerBlock,
        half_embedding: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:
        child_queries = (
            base_child_queries + half_embedding[None, None].to(base_child_queries.dtype)
        ).reshape(batch, p32_count * 2, dim)
        child_queries = child_queries * p16_valid.unsqueeze(-1).to(child_queries.dtype)
        child_state = _rooted_block(block, child_queries, state, token_valid)
        return norm(child_state * p16_valid.unsqueeze(-1).to(child_state.dtype))

    direction_hidden = tail(
        model.direction_tail,
        model.direction_half_embedding,
        model.direction_output_norm,
    )
    scale_hidden = tail(
        model.scale_tail,
        model.scale_half_embedding,
        model.scale_output_norm,
    )
    raw_direction = model.direction_head(direction_hidden).view(batch, 128, 8, 16)
    raw_scale = model.scale_head(scale_hidden).view(batch, 128, 8)
    pred_dirs = raw_direction.float() / (
        raw_direction.float().square().sum(dim=-1, keepdim=True) + 1.0e-12
    ).sqrt()
    pred_scales = model._bound_log_scale(raw_scale.float())
    return raw_direction, pred_dirs, pred_scales


def _direction_metrics(
    raw: torch.Tensor,
    dirs: torch.Tensor,
    scales: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    flat_dirs = dirs[valid].view(dirs.shape[0], -1)
    raw_norm = raw.float().square().sum(dim=-1).sqrt()[valid]
    return {
        "raw_rms": float(raw.float().square().mean().sqrt()),
        "raw_norm_median": float(raw_norm.median()),
        "direction_cross_sample_cosine": _pairwise_cosine_summary(flat_dirs),
        "direction_effective_rank": _effective_rank(flat_dirs),
        "scale_rms": float(scales.float().square().mean().sqrt()),
    }


def _change(base: torch.Tensor, changed: torch.Tensor, valid: torch.Tensor) -> float:
    cosine = (base.float() * changed.float()).sum(dim=-1)
    return float((1.0 - cosine[valid]).mean())


def _snapshot(
    model: PolarTailedUnifiedWeightBottleneck,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    W = batch["W"].to(device)
    X = batch["X"].to(device)
    x_mask = batch["x_mask"].to(device)
    d_in = batch["d_in_mask"].to(device)
    d_out = batch["d_out_mask"].to(device)
    tile_row = batch["tile_row"].to(device)
    tile_col = batch["tile_col"].to(device)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in,
        d_out,
        scale_mean=float(config["normalization"]["log2_scale_mean"]),
        scale_std=float(config["normalization"]["log2_scale_std"]),
    )
    content, log_scale, token_valid = (
        content.to(device),
        log_scale.to(device),
        token_valid.to(device),
    )
    component_mask = d_out[:, :, None, None] & d_in.view(W.shape[0], 8, 16)[:, None]
    p16_valid = component_mask.any(dim=-1).reshape(W.shape[0], -1)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        context = model._encode_distribution_context(X, sample_mask=x_mask)
        z, _ = model.encode(
            content,
            log_scale,
            tile_row,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=context,
        )
        _prediction, current_dirs, current_scales = model.decode_polar(
            z,
            tile_row,
            d_in_mask=d_in,
            d_out_mask=d_out,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=context,
        )
        rooted_raw, rooted_dirs, rooted_scales = _latent_rooted_decode(
            model, z, tile_row, tile_col, token_valid, p16_valid, context
        )
        permutation = torch.roll(torch.arange(z.shape[0], device=device), 1)
        _, rooted_roll, _ = _latent_rooted_decode(
            model, z[permutation], tile_row, tile_col, token_valid, p16_valid, context
        )
        zero_raw, rooted_zero, _ = _latent_rooted_decode(
            model, torch.zeros_like(z), tile_row, tile_col, token_valid, p16_valid, context
        )
        _, rooted_context_roll, _ = _latent_rooted_decode(
            model,
            z,
            tile_row,
            tile_col,
            token_valid,
            p16_valid,
            context[permutation],
        )
        _p, current_roll, _s = model.decode_polar(
            z[permutation],
            tile_row,
            d_in_mask=d_in,
            d_out_mask=d_out,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=context,
        )
        _p, current_zero, _s = model.decode_polar(
            torch.zeros_like(z),
            tile_row,
            d_in_mask=d_in,
            d_out_mask=d_out,
            tile_col=tile_col,
            token_valid_mask=token_valid,
            dist_patch_by_patch=context,
        )

    z_probe = z.detach().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, probe_dirs, _ = _latent_rooted_decode(
            model, z_probe, tile_row, tile_col, token_valid, p16_valid, context.detach()
        )
        probe = torch.Generator(device=device).manual_seed(1701)
        signs = torch.randint(
            0, 2, probe_dirs.shape, generator=probe, device=device, dtype=torch.int8
        ).float().mul_(2).sub_(1)
        scalar = (probe_dirs.float() * signs * component_mask).sum() / component_mask.sum()
    z_gradient = torch.autograd.grad(scalar, z_probe)[0]

    valid = component_mask.any(dim=-1)
    return {
        "latent": {
            "rms": float(z.float().square().mean().sqrt()),
            "effective_rank": _effective_rank(z.flatten(1)),
            "cross_sample_cosine": _pairwise_cosine_summary(z.flatten(1)),
        },
        "current": {
            "direction_cross_sample_cosine": _pairwise_cosine_summary(
                current_dirs[valid].view(W.shape[0], -1)
            ),
            "latent_roll_direction_change": _change(current_dirs, current_roll, valid),
            "latent_zero_direction_change": _change(current_dirs, current_zero, valid),
        },
        "latent_rooted": {
            **_direction_metrics(rooted_raw, rooted_dirs, rooted_scales, valid),
            "latent_roll_direction_change": _change(rooted_dirs, rooted_roll, valid),
            "latent_zero_direction_change": _change(rooted_dirs, rooted_zero, valid),
            "context_roll_direction_change": _change(rooted_dirs, rooted_context_roll, valid),
            "zero_raw_direction_max_abs": float(zero_raw.float().abs().max()),
            "latent_gradient_rms": float(z_gradient.float().square().mean().sqrt()),
            "latent_gradient_nonzero_fraction": float((z_gradient != 0).float().mean()),
        },
    }

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-update latent-rooted dependency audit."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    workspace = Path(__file__).resolve().parents[2]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (workspace / config_path).resolve()
    config = _load_config(config_path)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_stat = checkpoint_path.stat()
    print(f"[latent-rooted] stage=checkpoint path={checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    step = int(payload["step"])
    cursor = int(payload["committed_logical_index"])
    output_root = args.run_root.resolve() / f"latent_rooted_state_step_{step:07d}_v1"
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    batch = _load_batch(config, cursor, int(args.batch_size))
    device = torch.device("cuda:0")
    model_cfg = _model_config(config)
    snapshots: dict[str, Any] = {}
    for label in ("initial", "trained"):
        print(f"[latent-rooted] stage={label}", flush=True)
        _seed_everything(int(config["seed"]))
        model_cls = (
            LatentRootedPolarTailedUnifiedWeightBottleneck
            if config["schema"] == LATENT_ROOTED_POLAR_TAIL_SCHEMA
            else PolarTailedUnifiedWeightBottleneck
        )
        model = model_cls(model_cfg, "normalized_float").to(device)
        if label == "trained":
            model.load_state_dict(payload["model_state"], strict=True)
        snapshots[label] = _snapshot(model, batch, config)
        del model
        torch.cuda.empty_cache()
    report = {
        "schema": "weightclip_latent_rooted_decoder_preflight_v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
            "step": step,
            "cursor": cursor,
        },
        "batch": {"logical_indices": batch["logical_indices"]},
        "counterfactual": {
            "description": "Reuse frozen self-attention weights as Q<-K,V cross-attention; no query residual; no optimizer steps.",
            "limitations": "The modules were not initialized specifically for the changed graph; this tests causal dependency and numerical viability, not reconstruction quality.",
        },
        **snapshots,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / "report.json", report)
    summary = output_root / "summary.md"
    summary.write_text(
        "# Latent-rooted decoder zero-training preflight\n\n"
        f"Checkpoint step: {step}\n\n"
        "This is a frozen-weight graph counterfactual, not a quality evaluation.\n\n"
        + "```json\n"
        + json.dumps(snapshots, indent=2, sort_keys=True)
        + "\n```\n"
    )
    _atomic_json(output_root / "COMPLETE.json", {"complete": True, "report": str(output_root / "report.json")})
    print(f"[latent-rooted] stage=complete output={output_root} elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
