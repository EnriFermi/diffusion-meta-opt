from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from training.weightclip_benchmark.run_mini_polar_regression_production import (
    MiniPolarConfig,
    MiniPolarWeightBottleneck,
    SubtileCursor,
    _batch_from_stream,
    _load_config,
    _open_subtile_stream,
    _polar_routed_loss,
    _prepare_normalized_inputs,
)
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    _seed_everything,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-probe direction geometry and dependency audit for the mini polar run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--step10k", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().flatten()
    return {
        "mean": float(value.mean()),
        "p05": float(value.quantile(0.05)),
        "median": float(value.median()),
        "p95": float(value.quantile(0.95)),
    }


def _off_diagonal_cosine(value: torch.Tensor) -> float:
    value = value.detach().float().flatten(1)
    value = torch.nn.functional.normalize(value, dim=-1, eps=1.0e-8)
    cosine = value @ value.transpose(0, 1)
    count = value.shape[0]
    return float((cosine.sum() - cosine.diagonal().sum()) / max(count * (count - 1), 1))


def _effective_rank(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().flatten(1)
    value = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(value)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1.0e-20)
    entropy_rank = torch.exp(
        -(probability * probability.clamp_min(1.0e-20).log()).sum()
    )
    stable_rank = energy.sum() / energy.max().clamp_min(1.0e-20)
    return {
        "entropy_effective_rank": float(entropy_rank),
        "stable_rank": float(stable_rank),
        "top_energy_fraction": float(probability.max()),
    }


def _direction_change(
    reference: torch.Tensor,
    intervention: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    cosine = (reference.float() * intervention.float()).sum(dim=-1)
    cosine = cosine[valid]
    delta = (reference.float() - intervention.float()).square().sum(dim=-1).sqrt()
    delta = delta[valid]
    return {
        "mean_one_minus_cosine": float((1.0 - cosine).mean()),
        "mean_l2": float(delta.mean()),
    }


@torch.no_grad()
def _evaluate(
    label: str,
    model: MiniPolarWeightBottleneck,
    batch: dict[str, torch.Tensor],
    normalization: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    W = batch["W"].to(device)
    X = batch["X"].to(device)
    x_mask = batch["x_mask"].to(device)
    d_in_mask = batch["d_in_mask"].to(device)
    d_out_mask = batch["d_out_mask"].to(device)
    tile_row = batch["tile_row"].to(device)
    tile_col = batch["tile_col"].to(device)
    source_indices = batch["distribution_source_indices"].to(device)
    group_index = batch["distribution_group_index"].to(device)
    content, log_scale, token_valid = _prepare_normalized_inputs(
        W,
        d_in_mask,
        d_out_mask,
        scale_mean=float(normalization["log2_scale_mean"]),
        scale_std=float(normalization["log2_scale_std"]),
    )
    model.eval()
    raw_outputs: list[torch.Tensor] = []
    tail_outputs: list[torch.Tensor] = []
    raw_hook = model.direction_head.register_forward_hook(
        lambda _module, _args, output: raw_outputs.append(output.detach())
    )
    tail_hook = model.direction_tail.register_forward_hook(
        lambda _module, _args, output: tail_outputs.append(output.detach())
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dist_patch = model._encode_distribution_context(
            X,
            x_mask,
            source_indices,
            group_index,
        )
        latent, _ = model.encode(
            content,
            log_scale,
            tile_row,
            tile_col,
            token_valid,
            dist_patch,
        )
        prediction, pred_dirs, pred_scales = model.decode_polar(
            latent,
            tile_row,
            tile_col,
            d_in_mask,
            d_out_mask,
            token_valid,
            dist_patch,
        )
        loss, parts = _polar_routed_loss(
            X,
            W,
            pred_dirs,
            pred_scales,
            x_mask,
            d_in_mask,
            d_out_mask,
            config["loss"],
        )
    raw_hook.remove()
    tail_hook.remove()
    raw = raw_outputs[0].view(W.shape[0], 32, 2, 16).float()
    tail = tail_outputs[0][:, model.cfg.latent_slots :].float()
    valid = (
        d_out_mask[:, :, None]
        & d_in_mask.view(W.shape[0], 2, 16).any(dim=-1)[:, None, :]
    )
    component_mask = (
        d_out_mask[:, :, None, None]
        & d_in_mask.view(W.shape[0], 2, 16)[:, None, :, :]
    )
    target = W.transpose(1, 2).contiguous().view(W.shape[0], 32, 2, 16)
    target = target * component_mask.to(target.dtype)
    target_dirs = target / target.float().square().sum(dim=-1, keepdim=True).clamp_min(
        1.0e-12
    ).sqrt()

    shift = max(1, W.shape[0] // 2)
    interventions: dict[str, Any] = {}
    for intervention_name, current_latent, current_context in (
        ("latent_roll", latent.roll(shift, dims=0), dist_patch),
        ("distribution_context_roll", latent, dist_patch.roll(shift, dims=0)),
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, intervention_dirs, intervention_scales = model.decode_polar(
                current_latent,
                tile_row,
                tile_col,
                d_in_mask,
                d_out_mask,
                token_valid,
                current_context,
            )
            intervention_loss, intervention_parts = _polar_routed_loss(
                X,
                W,
                intervention_dirs,
                intervention_scales,
                x_mask,
                d_in_mask,
                d_out_mask,
                config["loss"],
            )
        interventions[intervention_name] = {
            **_direction_change(pred_dirs, intervention_dirs, valid),
            "total_loss_delta": float(intervention_loss - loss),
            "behavioral_direction_delta": float(
                intervention_parts["behavioral_direction"]
                - parts["behavioral_direction"]
            ),
            "structural_direction_delta": float(
                intervention_parts["structural_direction"]
                - parts["structural_direction"]
            ),
        }

    head = model.direction_head.weight.detach().float()
    singular = torch.linalg.svdvals(head)
    result = {
        "label": label,
        "loss": float(loss),
        "loss_parts": {key: float(value) for key, value in parts.items()},
        "direction_head": {
            "frobenius_norm": float(head.norm()),
            "top_singular": float(singular[0]),
            "stable_rank": float(head.square().sum() / singular[0].square()),
        },
        "raw_direction_logit_norm": _quantiles(raw.norm(dim=-1)[valid]),
        "direction_tail_state_rms": float(tail.square().mean().sqrt()),
        "prediction_target_patch_cosine": _quantiles(
            (pred_dirs.float() * target_dirs.float()).sum(dim=-1)[valid]
        ),
        "prediction_cross_sample_cosine": _off_diagonal_cosine(
            pred_dirs * component_mask.to(pred_dirs.dtype)
        ),
        "target_cross_sample_cosine": _off_diagonal_cosine(target_dirs),
        "latent_cross_sample_cosine": _off_diagonal_cosine(latent),
        "latent_rank": _effective_rank(latent),
        "interventions": interventions,
    }
    return result


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config.resolve())
    normalization = json.loads(
        Path(config["normalization"]["stats_path"]).read_text(encoding="utf-8")
    )
    logger = logging.getLogger("mini-direction-phase")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    print("[mini-direction-phase] stage=data-open", flush=True)
    with ExitStack() as stack:
        stream, _parent = _open_subtile_stream(
            stack,
            config,
            SubtileCursor(0, 0, 0),
            logger,
        )
        batch = _batch_from_stream(stream, int(args.batch_size))
    device = torch.device(str(config["device"]))
    reports: list[dict[str, Any]] = []
    checkpoints: list[tuple[str, Path | None]] = [
        ("seed42_initialization", None),
        ("step_10000", args.step10k.resolve()),
        ("post_transition", args.post.resolve()),
    ]
    for label, checkpoint_path in checkpoints:
        print(f"[mini-direction-phase] stage=evaluate label={label}", flush=True)
        _seed_everything(int(config["seed"]))
        model = MiniPolarWeightBottleneck(MiniPolarConfig(**config["architecture"]))
        checkpoint_step = 0
        if checkpoint_path is not None:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            checkpoint_step = int(payload["step"])
            model.load_state_dict(payload["model_state"], strict=True)
        model = model.to(device)
        report = _evaluate(
            label,
            model,
            batch,
            normalization,
            config,
            device,
        )
        report["checkpoint_step"] = checkpoint_step
        report["checkpoint_path"] = (
            None if checkpoint_path is None else str(checkpoint_path)
        )
        reports.append(report)
        del model
        torch.cuda.empty_cache()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mini_direction_phase_diagnostic_v1",
        "config": str(args.config.resolve()),
        "batch_size": int(args.batch_size),
        "fixed_probe_start": SubtileCursor(0, 0, 0).to_dict(),
        "reports": reports,
    }
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"[mini-direction-phase] stage=complete output={output}", flush=True)


if __name__ == "__main__":
    main()
