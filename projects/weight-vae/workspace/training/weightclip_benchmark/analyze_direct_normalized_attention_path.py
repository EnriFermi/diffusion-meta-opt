from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    DEFAULT_CONFIG,
    _load_config,
    _model_config,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    DEFAULT_RESOLVED,
    DEFAULT_SELECTION,
    UnifiedWeightBottleneck,
    _load_exact64_tiles,
    _seed_everything,
)


DEFAULT_CHECKPOINT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/"
    "direct_normalized_scaled_700m_p32_no_operator_500k_v1/model_latest.pt"
)
DEFAULT_OUTPUT = Path(
    "/mnt/shared/weightclip_benchmark/"
    "direct_normalized_scaled_700m_p32_no_operator_500k_v1/"
    "analysis_attention_path_latest.json"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_RESOLVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--production-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--initial", action="store_true")
    return parser.parse_args()


def _attention_components(
    block,
    state: torch.Tensor,
    key_context: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    batch, length, dim = state.shape
    qkv = block.qkv(block.attn_norm(state)).view(
        batch, length, 3, block.heads, block.head_dim
    )
    query, state_key, _value = qkv.unbind(dim=2)
    key = state_key
    context_key: torch.Tensor | None = None
    if block.context_key_projection is not None:
        if key_context is None:
            raise RuntimeError("missing encoder key context")
        normalized_context = F.rms_norm(
            key_context.float(), (key_context.shape[-1],), weight=None, eps=1.0e-6
        )
        context_key = block.context_key_projection(normalized_context).view(
            batch, length, block.heads, block.head_dim
        )
        key = key + context_key.to(key.dtype)
    query_heads = query.transpose(1, 2)
    key_heads = key.transpose(1, 2)
    if block.bounded_cosine_attention:
        query_heads = F.normalize(query_heads.float(), dim=-1, eps=1.0e-6)
        key_heads = F.normalize(key_heads.float(), dim=-1, eps=1.0e-6)
        scores = torch.einsum("bhld,bhsd->bhls", query_heads, key_heads)
        scores = scores * block.attention_logit_scale
    else:
        scores = torch.einsum("bhld,bhsd->bhls", query_heads.float(), key_heads.float())
        scores = scores / math.sqrt(block.head_dim)
    return (
        torch.softmax(scores, dim=-1),
        query,
        key,
        state_key,
        context_key,
        scores,
    )


def _attention_subset_diagnostics(
    *,
    attention: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    state_key: torch.Tensor,
    context_key: torch.Tensor | None,
    scores: torch.Tensor,
    query_start: int,
    query_stop: int,
    bounded_cosine_attention: bool,
    attention_logit_scale: float,
) -> dict[str, float]:
    query_subset = query[:, query_start:query_stop].transpose(1, 2).float()
    key_heads = key.transpose(1, 2).float()
    state_key_heads = state_key.transpose(1, 2).float()
    attention_subset = attention[:, :, query_start:query_stop].float()
    score_subset = scores[:, :, query_start:query_stop].float()

    if bounded_cosine_attention:
        state_scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            F.normalize(query_subset, dim=-1, eps=1.0e-6),
            F.normalize(state_key_heads, dim=-1, eps=1.0e-6),
        ) * attention_logit_scale
    else:
        state_scores = torch.einsum(
            "bhqd,bhkd->bhqk", query_subset, state_key_heads
        ) / math.sqrt(query_subset.shape[-1])
    state_attention = torch.softmax(state_scores, dim=-1)
    normalized_scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        F.normalize(query_subset, dim=-1),
        F.normalize(key_heads, dim=-1),
    ) * math.sqrt(query_subset.shape[-1])
    normalized_attention = torch.softmax(normalized_scores, dim=-1)

    argmax = attention_subset.argmax(dim=-1).permute(1, 2, 0).reshape(-1, attention.shape[0])
    mode_fractions = []
    for row in argmax:
        mode_fractions.append(
            torch.bincount(row, minlength=attention.shape[-1]).max().float()
            / attention.shape[0]
        )

    result = {
        "query_l2_mean": float(query_subset.norm(dim=-1).mean().item()),
        "state_key_l2_mean": float(state_key_heads.norm(dim=-1).mean().item()),
        "full_key_l2_mean": float(key_heads.norm(dim=-1).mean().item()),
        "logit_std": float(score_subset.std().item()),
        "logit_span_mean": float(
            (score_subset.amax(dim=-1) - score_subset.amin(dim=-1)).mean().item()
        ),
        "max_probability_mean": float(attention_subset.amax(dim=-1).mean().item()),
        "argmax_mode_fraction_across_batch": float(torch.stack(mode_fractions).mean().item()),
        "state_key_only_entropy_fraction": _entropy(state_attention),
        "state_key_only_max_probability_mean": float(
            state_attention.amax(dim=-1).mean().item()
        ),
        "qknorm_sqrtd_entropy_fraction": _entropy(normalized_attention),
        "qknorm_sqrtd_max_probability_mean": float(
            normalized_attention.amax(dim=-1).mean().item()
        ),
    }
    if context_key is not None:
        context_key_heads = context_key.transpose(1, 2).float()
        if bounded_cosine_attention:
            context_scores = torch.einsum(
                "bhqd,bhkd->bhqk",
                F.normalize(query_subset, dim=-1, eps=1.0e-6),
                F.normalize(context_key_heads, dim=-1, eps=1.0e-6),
            ) * attention_logit_scale
        else:
            context_scores = torch.einsum(
                "bhqd,bhkd->bhqk", query_subset, context_key_heads
            ) / math.sqrt(query_subset.shape[-1])
        context_attention = torch.softmax(context_scores, dim=-1)
        result.update(
            {
                "context_key_l2_mean": float(
                    context_key_heads.norm(dim=-1).mean().item()
                ),
                "context_key_only_entropy_fraction": _entropy(context_attention),
                "context_key_only_max_probability_mean": float(
                    context_attention.amax(dim=-1).mean().item()
                ),
            }
        )
    return result


def _block_forward_with_updates(
    block,
    state: torch.Tensor,
    key_context: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Replays the exact unmasked eval block while exposing both residual writes."""
    batch, length, dim = state.shape
    qkv = block.qkv(block.attn_norm(state)).view(
        batch, length, 3, block.heads, block.head_dim
    )
    query, key, value = qkv.unbind(dim=2)
    if block.context_key_projection is not None:
        if key_context is None:
            raise RuntimeError("missing encoder key context")
        normalized_context = F.rms_norm(
            key_context.float(), (key_context.shape[-1],), weight=None, eps=1.0e-6
        )
        context_key = block.context_key_projection(normalized_context).view(
            batch, length, block.heads, block.head_dim
        )
        key = key + context_key.to(key.dtype)
    elif key_context is not None:
        raise RuntimeError("unconditioned block received key context")
    query_heads = query.transpose(1, 2)
    key_heads = key.transpose(1, 2)
    if block.bounded_cosine_attention:
        query_heads = F.normalize(query_heads.float(), dim=-1, eps=1.0e-6).to(
            dtype=query.dtype
        )
        key_heads = F.normalize(key_heads.float(), dim=-1, eps=1.0e-6).to(
            dtype=key.dtype
        )
    attended = F.scaled_dot_product_attention(
        query_heads,
        key_heads,
        value.transpose(1, 2),
        dropout_p=0.0,
        scale=(block.attention_logit_scale if block.bounded_cosine_attention else None),
    )
    raw_attention_update = block.attn_out(
        attended.transpose(1, 2).reshape(batch, length, dim)
    )
    attention_update = block._bounded_write(raw_attention_update)
    after_attention = state + attention_update
    left, right = block.mlp_in(block.mlp_norm(after_attention)).chunk(2, dim=-1)
    raw_mlp_hidden = F.silu(left) * right
    mlp_hidden = raw_mlp_hidden
    if block.bounded_swiglu_hidden:
        hidden_rms_sq = mlp_hidden.float().square().mean(dim=-1, keepdim=True)
        mlp_hidden = mlp_hidden * torch.rsqrt(1.0 + hidden_rms_sq).to(
            dtype=mlp_hidden.dtype
        )
    raw_mlp_update = block.mlp_out(mlp_hidden)
    mlp_update = block._bounded_write(raw_mlp_update)
    return (
        after_attention + mlp_update,
        attention_update,
        mlp_update,
        mlp_hidden,
        raw_attention_update,
        raw_mlp_update,
        raw_mlp_hidden,
    )


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt().item())


def _relative_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        (a.float() - b.float()).square().mean().sqrt().div(
            a.float().square().mean().sqrt().clamp_min(1.0e-12)
        ).item()
    )


def _mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten(1), b.float().flatten(1), dim=1)
        .mean()
        .item()
    )


def _entropy(attention: torch.Tensor) -> float:
    entropy = -(attention * attention.clamp_min(1.0e-12).log()).sum(dim=-1)
    return float((entropy / math.log(attention.shape[-1])).mean().item())


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.initial:
        checkpoint_step = 0
        production_config = _load_config(args.production_config.resolve())
        checkpoint_label = "seed-42 initialization"
        model_state = None
    else:
        print(f"[attention-path] stage=load checkpoint={args.checkpoint}", flush=True)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_step = int(payload["step"])
        production_config = payload["config"]
        model_state = payload["model_state"]
        del payload
        checkpoint_label = str(args.checkpoint.resolve())
    cfg = _model_config(production_config)
    _seed_everything(cfg.seed)
    model = UnifiedWeightBottleneck(cfg, "normalized_float")
    if model_state is not None:
        model.load_state_dict(model_state, strict=True)
    data = _load_exact64_tiles(args.selection.resolve(), args.resolved_config.resolve())
    device = torch.device("cuda:0")
    model = model.to(device).eval()

    # Two operators, all nine aligned tiles. The paired roll preserves tile identity
    # and activation context while changing only the weight input / latent source.
    weights = data["weights"][:2].float().to(device)
    contexts = data["contexts"][:2].float().to(device)
    tile_rows = data["tile_rows"][:2].long().to(device)
    flat_weights = weights.flatten(0, 1)
    flat_contexts = contexts.flatten(0, 1)
    flat_rows = tile_rows.flatten(0, 1)
    flat_cols = torch.zeros_like(flat_rows)
    full_mask = torch.ones(18, 128, device=device, dtype=torch.bool)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        flat_weights,
        full_mask,
        full_mask,
        scale_mean=float(production_config["normalization"]["log2_scale_mean"]),
        scale_std=float(production_config["normalization"]["log2_scale_std"]),
    )
    shuffled_content = content.view(2, 9, *content.shape[1:]).roll(1, dims=0).reshape_as(content)
    shuffled_scale = log_scale.view(2, 9, *log_scale.shape[1:]).roll(1, dims=0).reshape_as(log_scale)

    group_ids = torch.arange(128, device=device).repeat_interleave(model.chunks_per_group)
    chunk_ids = torch.arange(model.chunks_per_group, device=device).repeat(128)

    def initial_encoder_state(current_content: torch.Tensor, current_scale: torch.Tensor) -> torch.Tensor:
        embedded = model._content_embedding(current_content)
        embedded = (
            embedded
            + model.scale_mlp(current_scale.float()).to(embedded.dtype)
            + model.group_embedding(group_ids)[None]
            + model.chunk_embedding(chunk_ids)[None]
            + model._tile_position_embedding(flat_rows, flat_cols)[:, None]
        )
        latent = model.latent_slots[None].expand(embedded.shape[0], -1, -1)
        return torch.cat((latent, embedded), dim=1)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dist = model._encode_distribution_context(flat_contexts)
        content_context = dist.unsqueeze(1).expand(-1, 128, -1, -1).reshape(
            18, model.weight_token_count, cfg.distribution_d_dist
        )
        latent_context = content_context.new_zeros(18, cfg.latent_slots, cfg.distribution_d_dist)
        key_context = torch.cat((latent_context, content_context), dim=1)
        state = initial_encoder_state(content, log_scale)
        shuffled_state = initial_encoder_state(shuffled_content, shuffled_scale)
        encoder_rows = []
        for depth, block in enumerate(model.encoder_blocks, start=1):
            attention, query, key, state_key, context_key, scores = (
                _attention_components(block, state, key_context)
            )
            latent_attention = attention[:, :, : cfg.latent_slots]
            attention_diagnostics = _attention_subset_diagnostics(
                attention=attention,
                query=query,
                key=key,
                state_key=state_key,
                context_key=context_key,
                scores=scores,
                query_start=0,
                query_stop=cfg.latent_slots,
                bounded_cosine_attention=block.bounded_cosine_attention,
                attention_logit_scale=block.attention_logit_scale,
            )
            input_latent = state[:, : cfg.latent_slots]
            (
                state,
                attention_update,
                mlp_update,
                mlp_hidden,
                raw_attention_update,
                raw_mlp_update,
                raw_mlp_hidden,
            ) = _block_forward_with_updates(block, state, key_context)
            (
                shuffled_state,
                shuffled_attention_update,
                shuffled_mlp_update,
                _,
                _,
                _,
                _,
            ) = _block_forward_with_updates(block, shuffled_state, key_context)
            latent_attention_update = attention_update[:, : cfg.latent_slots]
            latent_mlp_update = mlp_update[:, : cfg.latent_slots]
            shuffled_latent_attention_update = shuffled_attention_update[
                :, : cfg.latent_slots
            ]
            shuffled_latent_mlp_update = shuffled_mlp_update[:, : cfg.latent_slots]
            encoder_rows.append(
                {
                    "depth": depth,
                    "latent_query_mass_to_weight_tokens": float(
                        latent_attention[..., cfg.latent_slots :].sum(dim=-1).mean().item()
                    ),
                    "latent_query_attention_entropy_fraction": _entropy(latent_attention),
                    "latent_input_rms": _rms(input_latent),
                    "latent_attention_update_rms": _rms(latent_attention_update),
                    "latent_attention_update_raw_rms": _rms(
                        raw_attention_update[:, : cfg.latent_slots]
                    ),
                    "latent_attention_update_cosine_to_input": _mean_cosine(
                        latent_attention_update, input_latent
                    ),
                    "latent_attention_update_weight_shuffle_cosine": _mean_cosine(
                        latent_attention_update, shuffled_latent_attention_update
                    ),
                    "latent_attention_update_weight_shuffle_relative_delta": _relative_delta(
                        latent_attention_update, shuffled_latent_attention_update
                    ),
                    "latent_mlp_update_rms": _rms(latent_mlp_update),
                    "latent_mlp_update_raw_rms": _rms(
                        raw_mlp_update[:, : cfg.latent_slots]
                    ),
                    "latent_mlp_hidden_rms": _rms(
                        mlp_hidden[:, : cfg.latent_slots]
                    ),
                    "latent_mlp_hidden_raw_rms": _rms(
                        raw_mlp_hidden[:, : cfg.latent_slots]
                    ),
                    "latent_mlp_update_cosine_to_input": _mean_cosine(
                        latent_mlp_update, input_latent
                    ),
                    "latent_mlp_update_weight_shuffle_cosine": _mean_cosine(
                        latent_mlp_update, shuffled_latent_mlp_update
                    ),
                    "latent_mlp_update_weight_shuffle_relative_delta": _relative_delta(
                        latent_mlp_update, shuffled_latent_mlp_update
                    ),
                }
            )
            current_latent = state[:, : cfg.latent_slots]
            shuffled_latent = shuffled_state[:, : cfg.latent_slots]
            encoder_rows[-1].update(
                {
                    "latent_weight_shuffle_relative_delta": _relative_delta(
                        current_latent, shuffled_latent
                    ),
                    "latent_weight_shuffle_cosine": _mean_cosine(
                        current_latent, shuffled_latent
                    ),
                    "latent_state_rms": float(
                        current_latent.float().square().mean().sqrt().item()
                    ),
                }
            )
            encoder_rows[-1].update(attention_diagnostics)
        z = model.to_latent(model.latent_norm(state[:, : cfg.latent_slots]))
        shuffled_z = model.to_latent(
            model.latent_norm(shuffled_state[:, : cfg.latent_slots])
        )

        def initial_decoder_state(current_z: torch.Tensor) -> torch.Tensor:
            latent = model.from_latent(current_z)
            queries = model.output_queries[None].expand(current_z.shape[0], -1, -1)
            queries = queries + model._tile_position_embedding(flat_rows, flat_cols)[:, None]
            decoder_context = dist.unsqueeze(1).expand(-1, 128, -1, -1).reshape(
                18, model.weight_token_count, cfg.distribution_d_dist
            )
            queries = model.decoder_query_conditioner(
                torch.cat((queries, decoder_context.to(queries.dtype)), dim=-1)
            )
            return torch.cat((latent, queries), dim=1)

        decoder_state = initial_decoder_state(z)
        shuffled_decoder_state = initial_decoder_state(shuffled_z)
        decoder_rows = []
        for depth, block in enumerate(model.decoder_blocks, start=1):
            attention, query, key, state_key, context_key, scores = (
                _attention_components(block, decoder_state, None)
            )
            output_query_attention = attention[:, :, cfg.latent_slots :]
            attention_diagnostics = _attention_subset_diagnostics(
                attention=attention,
                query=query,
                key=key,
                state_key=state_key,
                context_key=context_key,
                scores=scores,
                query_start=cfg.latent_slots,
                query_stop=decoder_state.shape[1],
                bounded_cosine_attention=block.bounded_cosine_attention,
                attention_logit_scale=block.attention_logit_scale,
            )
            input_queries = decoder_state[:, cfg.latent_slots :]
            (
                decoder_state,
                attention_update,
                mlp_update,
                mlp_hidden,
                raw_attention_update,
                raw_mlp_update,
                raw_mlp_hidden,
            ) = _block_forward_with_updates(block, decoder_state, None)
            (
                shuffled_decoder_state,
                _shuffled_attention_update,
                _shuffled_mlp_update,
                _,
                _,
                _,
                _,
            ) = _block_forward_with_updates(block, shuffled_decoder_state, None)
            query_attention_update = attention_update[:, cfg.latent_slots :]
            query_mlp_update = mlp_update[:, cfg.latent_slots :]
            decoder_rows.append(
                {
                    "depth": depth,
                    "output_query_mass_to_latent_tokens": float(
                        output_query_attention[..., : cfg.latent_slots]
                        .sum(dim=-1)
                        .mean()
                        .item()
                    ),
                    "output_query_attention_entropy_fraction": _entropy(
                        output_query_attention
                    ),
                    "output_query_input_rms": _rms(input_queries),
                    "output_query_attention_update_rms": _rms(query_attention_update),
                    "output_query_attention_update_raw_rms": _rms(
                        raw_attention_update[:, cfg.latent_slots :]
                    ),
                    "output_query_attention_update_cosine_to_input": _mean_cosine(
                        query_attention_update, input_queries
                    ),
                    "output_query_mlp_update_rms": _rms(query_mlp_update),
                    "output_query_mlp_update_raw_rms": _rms(
                        raw_mlp_update[:, cfg.latent_slots :]
                    ),
                    "output_query_mlp_hidden_rms": _rms(
                        mlp_hidden[:, cfg.latent_slots :]
                    ),
                    "output_query_mlp_hidden_raw_rms": _rms(
                        raw_mlp_hidden[:, cfg.latent_slots :]
                    ),
                    "output_query_mlp_update_cosine_to_input": _mean_cosine(
                        query_mlp_update, input_queries
                    ),
                }
            )
            current_queries = decoder_state[:, cfg.latent_slots :]
            shuffled_queries = shuffled_decoder_state[:, cfg.latent_slots :]
            decoder_rows[-1].update(
                {
                    "output_query_weight_shuffle_relative_delta": _relative_delta(
                        current_queries, shuffled_queries
                    ),
                    "output_query_weight_shuffle_cosine": _mean_cosine(
                        current_queries, shuffled_queries
                    ),
                    "output_query_state_rms": float(
                        current_queries.float().square().mean().sqrt().item()
                    ),
                }
            )
            decoder_rows[-1].update(attention_diagnostics)

        output = model.output_head(
            model.output_norm(decoder_state[:, cfg.latent_slots :])
        )
        shuffled_output = model.output_head(
            model.output_norm(shuffled_decoder_state[:, cfg.latent_slots :])
        )

    result = {
        "schema": "direct_normalized_attention_path_v1",
        "checkpoint": checkpoint_label,
        "checkpoint_step": checkpoint_step,
        "uniform_encoder_latent_to_weight_mass": model.weight_token_count
        / (model.weight_token_count + cfg.latent_slots),
        "uniform_decoder_output_to_latent_mass": cfg.latent_slots
        / (model.weight_token_count + cfg.latent_slots),
        "encoder": encoder_rows,
        "decoder": decoder_rows,
        "final_latent_weight_shuffle_relative_delta": _relative_delta(z, shuffled_z),
        "final_latent_weight_shuffle_cosine": _mean_cosine(z, shuffled_z),
        "final_output_weight_shuffle_relative_delta": _relative_delta(output, shuffled_output),
        "final_output_weight_shuffle_cosine": _mean_cosine(output, shuffled_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[attention-path] stage=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
