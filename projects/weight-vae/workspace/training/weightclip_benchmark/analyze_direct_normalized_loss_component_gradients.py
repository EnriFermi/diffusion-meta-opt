from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    BOUNDED_BALANCED_SCHEMA,
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
    "analysis_checkpoint_060000_loss_component_gradients.json"
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


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.float().norm().item())


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().square().mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()
    )


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    production_config = _load_config(args.production_config.resolve())
    cfg = _model_config(production_config)
    _seed_everything(cfg.seed)
    model = UnifiedWeightBottleneck(cfg, "normalized_float")
    if args.initial:
        checkpoint_step = 0
        checkpoint_label = "seed-42 initialization"
    else:
        print(f"[component-grad] stage=load checkpoint={args.checkpoint}", flush=True)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_step = int(payload["step"])
        production_config = payload["config"]
        model.load_state_dict(payload["model_state"], strict=True)
        del payload
        checkpoint_label = str(args.checkpoint.resolve())

    print("[component-grad] stage=load-exact64 operators=2 tiles=18", flush=True)
    data = _load_exact64_tiles(args.selection.resolve(), args.resolved_config.resolve())
    weights = data["weights"][:2].float().flatten(0, 1)
    contexts = data["contexts"][:2].float().flatten(0, 1)
    tile_rows = data["tile_rows"][:2].long().flatten(0, 1)
    tile_cols = torch.zeros_like(tile_rows)
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    weights = weights.to(device)
    contexts = contexts.to(device)
    tile_rows = tile_rows.to(device)
    tile_cols = tile_cols.to(device)
    full_mask = torch.ones(18, 128, device=device, dtype=torch.bool)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        weights,
        full_mask,
        full_mask,
        scale_mean=float(production_config["normalization"]["log2_scale_mean"]),
        scale_std=float(production_config["normalization"]["log2_scale_std"]),
    )

    print("[component-grad] stage=forward-backward", flush=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction, _z, _ = model(
            content,
            log_scale,
            tile_rows,
            contexts,
            tile_col=tile_cols,
            token_valid_mask=token_valid,
        )
        behavioral_direction, behavioral_scale = (
            BigWeightVAELossMixin.operator_direction_scale_loss(
                contexts,
                weights,
                prediction,
                gamma=0.5,
                huber_delta=0.1,
            )
        )
        structural_direction, _ = BigWeightVAELossMixin.patch_structure_loss(
            weights,
            prediction,
            patch_size=16,
            gamma=0.5,
            lambda_dir=1.0,
            lambda_scale=0.0,
            lambda_rec=0.0,
            lambda_rel=0.0,
            huber_delta=0.1,
        )
        structural_scale, _ = BigWeightVAELossMixin.patch_structure_loss(
            weights,
            prediction,
            patch_size=16,
            gamma=0.5,
            lambda_dir=0.0,
            lambda_scale=1.0,
            lambda_rec=0.0,
            lambda_rel=0.0,
            huber_delta=0.1,
        )
        direction_loss = behavioral_direction + structural_direction
        scale_multiplier = (
            1.0
            if production_config["schema"] == BOUNDED_BALANCED_SCHEMA
            else 10.0
        )
        scale_loss = scale_multiplier * (behavioral_scale + structural_scale)

    direction_latent_grad = torch.autograd.grad(
        direction_loss, _z, retain_graph=True, allow_unused=False
    )[0].float()
    scale_latent_grad = torch.autograd.grad(
        scale_loss, _z, retain_graph=True, allow_unused=False
    )[0].float()
    direction_latent_rms = direction_latent_grad.square().mean().sqrt()
    scale_latent_rms = scale_latent_grad.square().mean().sqrt()
    normalized_latent_sum = (
        direction_latent_grad / direction_latent_rms
        + scale_latent_grad / scale_latent_rms
    )
    normalized_latent_sum_rms = normalized_latent_sum.square().mean().sqrt()
    raw_latent_sum_rms = (
        direction_latent_grad + scale_latent_grad
    ).square().mean().sqrt()
    direction_dynamic_weight = raw_latent_sum_rms / (
        normalized_latent_sum_rms * direction_latent_rms
    )
    scale_dynamic_weight = raw_latent_sum_rms / (
        normalized_latent_sum_rms * scale_latent_rms
    )
    production_loss = (
        direction_dynamic_weight.detach() * direction_loss
        + scale_dynamic_weight.detach() * scale_loss
    )

    named_parameters: list[tuple[str, torch.nn.Parameter]] = []
    for depth, block in enumerate(model.encoder_blocks, start=1):
        named_parameters.append((f"encoder_{depth:02d}.qkv", block.qkv.weight))
    named_parameters.extend(
        [
            ("encoder_01.context_key", model.encoder_blocks[0].context_key_projection.weight),
            ("encoder_01.attn_out", model.encoder_blocks[0].attn_out.weight),
            ("encoder_01.mlp_in", model.encoder_blocks[0].mlp_in.weight),
            ("encoder_01.mlp_out", model.encoder_blocks[0].mlp_out.weight),
            ("to_latent", model.to_latent.weight),
            ("decoder_01.qkv", model.decoder_blocks[0].qkv.weight),
            ("output_head", model.output_head.weight),
        ]
    )
    parameters = tuple(parameter for _name, parameter in named_parameters)
    direction_gradients = torch.autograd.grad(
        direction_loss, parameters, retain_graph=True, allow_unused=False
    )
    scale_gradients = torch.autograd.grad(
        scale_loss, parameters, retain_graph=True, allow_unused=False
    )
    production_gradients = torch.autograd.grad(
        production_loss, parameters, retain_graph=False, allow_unused=False
    )

    rows: dict[str, dict[str, float]] = {}
    dim = cfg.hidden_dim
    for (name, parameter), direction_grad, scale_grad, production_grad in zip(
        named_parameters,
        direction_gradients,
        scale_gradients,
        production_gradients,
        strict=True,
    ):
        slices = {"all": slice(None)}
        if name.endswith(".qkv"):
            slices = {
                "q": slice(0, dim),
                "k": slice(dim, 2 * dim),
                "v": slice(2 * dim, 3 * dim),
            }
        for suffix, current_slice in slices.items():
            direction_slice = direction_grad[current_slice]
            scale_slice = scale_grad[current_slice]
            production_slice = production_grad[current_slice]
            parameter_slice = parameter[current_slice]
            key = f"{name}.{suffix}"
            rows[key] = {
                "direction_norm": _norm(direction_slice),
                "direction_rms": _rms(direction_slice),
                "scale_norm": _norm(scale_slice),
                "scale_rms": _rms(scale_slice),
                "scale_over_direction_norm": _norm(scale_slice)
                / max(_norm(direction_slice), 1.0e-30),
                "balanced_scale_over_direction_norm": (
                    float(scale_dynamic_weight.item()) * _norm(scale_slice)
                    / max(
                        float(direction_dynamic_weight.item()) * _norm(direction_slice),
                        1.0e-30,
                    )
                ),
                "direction_scale_cosine": _cosine(direction_slice, scale_slice),
                "direction_weight_cosine": _cosine(direction_slice, parameter_slice),
                "scale_weight_cosine": _cosine(scale_slice, parameter_slice),
                "scale_descent_norm_growth_alignment": -_cosine(
                    scale_slice, parameter_slice
                ),
                "production_norm": _norm(production_slice),
                "production_weight_cosine": _cosine(
                    production_slice, parameter_slice
                ),
                "proxy_production_cosine": _cosine(
                    direction_slice + scale_slice, production_slice
                ),
                "balanced_proxy_production_cosine": _cosine(
                    float(direction_dynamic_weight.item()) * direction_slice
                    + float(scale_dynamic_weight.item()) * scale_slice,
                    production_slice,
                ),
            }

    encoder_q = [rows[f"encoder_{depth:02d}.qkv.q"] for depth in range(1, 14)]
    encoder_k = [rows[f"encoder_{depth:02d}.qkv.k"] for depth in range(1, 14)]
    result = {
        "schema": "direct_normalized_loss_component_gradients_v1",
        "checkpoint": checkpoint_label,
        "checkpoint_step": checkpoint_step,
        "batch": {"operators": 2, "tiles": 18},
        "losses": {
            "behavioral_direction": float(behavioral_direction.detach().float().item()),
            "behavioral_scale_raw": float(behavioral_scale.detach().float().item()),
            "structural_direction": float(structural_direction.detach().float().item()),
            "structural_scale_raw": float(structural_scale.detach().float().item()),
            "direction_total": float(direction_loss.detach().float().item()),
            "scale_total_weighted": float(scale_loss.detach().float().item()),
            "production_total": float(production_loss.detach().float().item()),
        },
        "bottleneck_balance": {
            "direction_grad_rms": float(direction_latent_rms.item()),
            "scale_grad_rms": float(scale_latent_rms.item()),
            "direction_scale_cosine": _cosine(
                direction_latent_grad, scale_latent_grad
            ),
            "direction_dynamic_weight": float(direction_dynamic_weight.item()),
            "scale_dynamic_weight": float(scale_dynamic_weight.item()),
        },
        "rows": rows,
        "encoder_q_summary": {
            "scale_over_direction_min": min(
                row["scale_over_direction_norm"] for row in encoder_q
            ),
            "scale_over_direction_median": float(
                torch.tensor(
                    [row["scale_over_direction_norm"] for row in encoder_q]
                ).median().item()
            ),
            "scale_over_direction_max": max(
                row["scale_over_direction_norm"] for row in encoder_q
            ),
            "direction_scale_cosine_min": min(
                row["direction_scale_cosine"] for row in encoder_q
            ),
            "direction_scale_cosine_median": float(
                torch.tensor([row["direction_scale_cosine"] for row in encoder_q])
                .median()
                .item()
            ),
            "direction_scale_cosine_max": max(
                row["direction_scale_cosine"] for row in encoder_q
            ),
        },
        "encoder_k_summary": {
            "scale_over_direction_min": min(
                row["scale_over_direction_norm"] for row in encoder_k
            ),
            "scale_over_direction_median": float(
                torch.tensor(
                    [row["scale_over_direction_norm"] for row in encoder_k]
                ).median().item()
            ),
            "scale_over_direction_max": max(
                row["scale_over_direction_norm"] for row in encoder_k
            ),
            "direction_scale_cosine_min": min(
                row["direction_scale_cosine"] for row in encoder_k
            ),
            "direction_scale_cosine_median": float(
                torch.tensor([row["direction_scale_cosine"] for row in encoder_k])
                .median()
                .item()
            ),
            "direction_scale_cosine_max": max(
                row["direction_scale_cosine"] for row in encoder_k
            ),
        },
    }
    for row in rows.values():
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError("nonfinite component gradient metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[component-grad] stage=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
