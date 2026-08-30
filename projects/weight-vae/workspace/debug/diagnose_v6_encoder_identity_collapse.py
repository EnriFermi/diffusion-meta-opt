from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer
from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from training.big_vae.data_types import _flatten_loader_batches, _identity_sample_collate
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.big_vae.two_operator_overfit import two_operator_data_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _tensor_output(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    return None


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    return F.cosine_similarity(a, b, dim=1, eps=1.0e-30)


def paired_stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten(1)
    a, b = flat[0::2], flat[1::2]
    delta = a - b
    pair_scale = (0.5 * (a.square().mean(1) + b.square().mean(1))).sqrt().clamp_min(1.0e-30)
    common = 0.5 * (a + b)
    difference = 0.5 * delta
    batch_centered = flat - flat.mean(0, keepdim=True)
    output = {
        "rms": _rms(flat),
        "paired_delta_rms": _rms(delta),
        "paired_relative_rms_mean": float((delta.square().mean(1).sqrt() / pair_scale).mean().item()),
        "paired_cosine_mean": float(F.cosine_similarity(a, b, dim=1, eps=1.0e-30).mean().item()),
        "pair_common_energy_fraction_mean": float(
            (common.square().sum(1) / (common.square().sum(1) + difference.square().sum(1)).clamp_min(1.0e-30)).mean().item()
        ),
        "batch_centered_energy_fraction": float(
            (batch_centered.square().sum() / flat.square().sum().clamp_min(1.0e-30)).item()
        ),
    }
    if value.ndim == 3:
        centered = value.detach().float() - value.detach().float().mean(1, keepdim=True)
        output["sequence_centered_energy_fraction"] = float(
            (centered.square().sum() / value.detach().float().square().sum().clamp_min(1.0e-30)).item()
        )
    return output


def comparison_stats(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float()
    value = value.detach().float()
    delta = value - reference
    return {
        "delta_rms": _rms(delta),
        "relative_rms": float((_rms(delta) / max(_rms(reference), 1.0e-30))),
        "cosine_mean": float(_cosine(reference, value).mean().item()),
    }


def direction_loss(target: torch.Tensor, prediction: torch.Tensor, cfg: Any) -> torch.Tensor:
    total, _ = WeightQuantileVAE.patch_structure_loss(
        target,
        prediction,
        patch_size=int(cfg.model.patch_size),
        gamma=float(cfg.train.struct_loss.gamma),
        lambda_dir=1.0,
        lambda_scale=0.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=float(cfg.train.struct_loss.huber_delta),
    )
    return total


def add_hook(module: torch.nn.Module, name: str, active: dict[str, torch.Tensor], handles: list[Any]) -> None:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        value = _tensor_output(output)
        if value is not None and value.ndim > 0:
            active[name] = value

    handles.append(module.register_forward_hook(hook))


def register_encoder_hooks(model: torch.nn.Module, active: dict[str, torch.Tensor]) -> list[Any]:
    handles: list[Any] = []
    add_hook(model.patch_tokenizer.y_proj, "tokenizer.y_proj", active, handles)
    add_hook(model.patch_tokenizer, "tokenizer.output", active, handles)
    add_hook(model.patch_token_proj, "tokenizer.patch_token_proj", active, handles)
    if model.latent_to_weight_feedback is not None:
        add_hook(model.latent_to_weight_feedback, "encoder.feedback.output", active, handles)
    for index, layer in enumerate(model.encoder_layers):
        prefix = f"encoder.{index}"
        add_hook(layer.local_block, f"{prefix}.local_output", active, handles)
        block = layer.perceiver_block
        add_hook(block.cross_out_proj, f"{prefix}.cross_projection", active, handles)
        add_hook(block.cross_return_norm, f"{prefix}.cross_return_normalized", active, handles)
        add_hook(block.self_out_proj, f"{prefix}.self_projection", active, handles)
        add_hook(block.self_return_norm, f"{prefix}.self_return_normalized", active, handles)
        add_hook(block.ffn[-2], f"{prefix}.ffn_projection", active, handles)
        add_hook(block.ffn_return_norm, f"{prefix}.ffn_return_normalized", active, handles)
        add_hook(block, f"{prefix}.latent_output", active, handles)
    add_hook(model.mandatory_latent_bridge.k_proj, "decoder.bridge_k", active, handles)
    add_hook(model.mandatory_latent_bridge.v_proj, "decoder.bridge_v", active, handles)
    add_hook(model.mandatory_latent_bridge, "decoder.bridge_output", active, handles)
    add_hook(model.position_only_film_v6, "decoder.posfilm_output", active, handles)
    add_hook(model.q_tokens_norm, "decoder.final_qnorm", active, handles)
    add_hook(model.direction_head, "decoder.direction_head", active, handles)
    return handles


def load_batch(cfg: Any, logger: logging.Logger) -> Any:
    two_cfg = cfg.train.operator_bank.two_operator_overfit
    with two_operator_data_pipeline(
        cfg.train.operator_bank.pair_manifest,
        selected_operators=OmegaConf.to_container(two_cfg.selected_operators, resolve=True),
        seed=int(cfg.data.seed),
        hot_shards=int(cfg.train.operator_bank.hot_shards),
        expected_pair_manifest_sha256=str(cfg.train.operator_bank.pair_manifest_sha256),
        max_active_strata=int(cfg.train.operator_bank.max_active_strata),
        max_active_bundle_bytes=int(cfg.train.operator_bank.max_active_bundle_bytes),
    ) as (dataset, sampler):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, sampler=sampler, num_workers=0, collate_fn=_identity_sample_collate
        )
        mixer = BalancedOperatorBankMixer(dataset, _flatten_loader_batches(iter(loader)), start_index=0)
        batch = _fetch_presliced_training_batch_cpu(dataset_iter=mixer, batch_size=18, logger=logger)
    if tuple(batch.logical_indices) != tuple(range(18)):
        raise RuntimeError(f"unexpected logical indices: {batch.logical_indices}")
    return batch


def run_forward(
    model: torch.nn.Module,
    W: torch.Tensor,
    X: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    *,
    grad: bool,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    active: dict[str, torch.Tensor] = {}
    handles = register_encoder_hooks(model, active)
    context = contextlib.nullcontext() if grad else torch.no_grad()
    try:
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            W_hat, _mu, _logvar, _dirs, debug = model.forward_debug(
                W, X, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
            )
            active["encoder.raw_z"] = debug["latent_decoder_z"]
            active["encoder.final_patch_tokens"] = debug["encoder_patch_tokens"]
            active["decoder.output_W_hat"] = W_hat
    finally:
        for handle in handles:
            handle.remove()
    return W_hat, debug, active


@contextlib.contextmanager
def zero_parameters(parameters: Iterable[torch.nn.Parameter]) -> Any:
    selected = list(dict.fromkeys(parameters))
    saved = [parameter.detach().clone() for parameter in selected]
    with torch.no_grad():
        for parameter in selected:
            parameter.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(selected, saved):
                parameter.copy_(value)


def residual_return_biases(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    biases: list[torch.nn.Parameter] = []
    for block in model.patch_tokenizer.blocks:
        biases.extend([block.base_fc2.bias, block.cond_fc2.bias])
    for layer in model.encoder_layers:
        biases.extend(
            [
                layer.local_block.out_proj.bias,
                layer.local_block.ffn.fc2.bias,
                layer.perceiver_block.cross_out_proj.bias,
                layer.perceiver_block.self_out_proj.bias,
                layer.perceiver_block.ffn[-2].bias,
            ]
        )
    if model.latent_to_weight_feedback is not None:
        biases.append(model.latent_to_weight_feedback.out.bias)
    if model.encoder_conditioning_adapters is not None:
        biases.extend(adapter.mix_out.bias for adapter in model.encoder_conditioning_adapters)
    return [bias for bias in biases if bias is not None]


def tokenizer_all_biases(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for name, parameter in model.patch_tokenizer.named_parameters() if name.endswith("bias")]


def gradient_cancellation(
    W: torch.Tensor,
    W_hat: torch.Tensor,
    z: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    cfg: Any,
) -> dict[str, Any]:
    rows: dict[str, list[torch.Tensor]] = {"raw_z": []}
    rows.update({name: [] for name in parameters})
    targets = [z, *parameters.values()]
    names = ["raw_z", *parameters]
    for index in range(int(W.shape[0])):
        loss = direction_loss(W[index : index + 1], W_hat[index : index + 1], cfg)
        grads = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
        for name, grad in zip(names, grads):
            if grad is None:
                rows[name].append(torch.zeros(1, device=W.device))
            elif name == "raw_z":
                rows[name].append(grad[index].detach().float().flatten())
            else:
                rows[name].append(grad.detach().float().flatten())
    output: dict[str, Any] = {}
    for name, vectors in rows.items():
        matrix = torch.stack(vectors)
        norms = matrix.norm(dim=1)
        normalized = matrix / norms.unsqueeze(1).clamp_min(1.0e-30)
        pairwise = normalized @ normalized.T
        mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
        output[name] = {
            "individual_grad_rms_mean": float(matrix.square().mean(1).sqrt().mean().item()),
            "individual_grad_l2_mean": float(norms.mean().item()),
            "sum_over_sum_norms": float((matrix.sum(0).norm() / norms.sum().clamp_min(1.0e-30)).item()),
            "pairwise_cosine_offdiag_mean": float(pairwise[mask].mean().item()),
            "paired_A_B_cosine_mean": float((normalized[0::2] * normalized[1::2]).sum(1).mean().item()),
        }
    return output


def decode_metrics(
    model: torch.nn.Module,
    W: torch.Tensor,
    debug: dict[str, Any],
    z: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    cfg: Any,
) -> dict[str, float]:
    swap = torch.arange(W.shape[0], device=W.device) ^ 1
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        matched, *_ = model._decode_from_decoder_latent(
            z,
            dist_patch_by_patch=debug["dist_patch_by_patch"],
            patch_mask=debug["patch_mask"],
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            d_in=int(W.shape[1]),
            d_out=int(W.shape[2]),
            d_in_pad=int(debug["d_in_pad"]),
            T=int(debug["T"]),
        )
        swapped, *_ = model._decode_from_decoder_latent(
            z.index_select(0, swap),
            dist_patch_by_patch=debug["dist_patch_by_patch"],
            patch_mask=debug["patch_mask"],
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            d_in=int(W.shape[1]),
            d_out=int(W.shape[2]),
            d_in_pad=int(debug["d_in_pad"]),
            T=int(debug["T"]),
        )
        output: dict[str, float] = {}
        for operator, indices in (("A", torch.arange(0, 18, 2, device=W.device)), ("B", torch.arange(1, 18, 2, device=W.device))):
            matched_loss = direction_loss(W.index_select(0, indices), matched.index_select(0, indices), cfg)
            swapped_loss = direction_loss(W.index_select(0, indices), swapped.index_select(0, indices), cfg)
            output[f"{operator}_matched_dir"] = float(matched_loss.item())
            output[f"{operator}_swap_gap"] = float((swapped_loss - matched_loss).item())
        output["matched_mean_dir"] = 0.5 * (output["A_matched_dir"] + output["B_matched_dir"])
        output["swap_min_gap"] = min(output["A_swap_gap"], output["B_swap_gap"])
    return output


def y_projection_contributions(model: torch.nn.Module, W: torch.Tensor) -> dict[str, float]:
    p = int(model.cfg.patch_size)
    B, d_in, d_out = W.shape
    w_patch = W.transpose(1, 2).reshape(B * d_out * (d_in // p), p).float()
    layer = model.patch_tokenizer.y_proj
    weight_signal = F.linear(w_patch.unsqueeze(-1), layer.weight.detach().float(), None)
    bias_signal = layer.bias.detach().float().view(1, 1, -1).expand_as(weight_signal)
    total = weight_signal + bias_signal
    return {
        "W_rms": _rms(W),
        "y_proj_weight_rms": _rms(layer.weight),
        "y_proj_bias_rms": _rms(layer.bias),
        "weight_contribution_rms": _rms(weight_signal),
        "bias_contribution_rms": _rms(bias_signal),
        "total_rms": _rms(total),
        "bias_to_weight_contribution_ratio": _rms(bias_signal) / max(_rms(weight_signal), 1.0e-30),
    }


def activation_report(active: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: paired_stats(value) for name, value in active.items() if int(value.shape[0]) == 18}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "progress.log")],
    )
    logger = logging.getLogger("v6-encoder-identity")
    start = time.time()
    logger.info("stage=start checkpoint=%s config=%s output=%s optimizer_steps=0", args.checkpoint, args.config, output_dir)
    cfg = OmegaConf.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model_state"]
    step = int(checkpoint["step"])
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    device = torch.device(args.device)
    model.to(device).eval()
    logger.info("stage=model_ready step=%s params=%s device=%s dtype=bf16", step, sum(p.numel() for p in model.parameters()), device)

    batch = load_batch(cfg, logger)
    W = batch.W.to(device)
    X = batch.x.to(device)
    x_mask = batch.x_mask.to(device)
    d_in_mask = batch.d_in_mask.to(device)
    d_out_mask = batch.d_out_mask.to(device)
    swap = torch.arange(18, device=device) ^ 1
    logger.info("stage=data_ready W=%s X=%s", tuple(W.shape), tuple(X.shape))

    W_hat, debug, matched = run_forward(model, W, X, x_mask, d_in_mask, d_out_mask, grad=True)
    z_flat = debug["latent_decoder_z"]
    z = z_flat.view(18, int(cfg.model.big_vae.num_latents), int(cfg.model.big_vae.d_lat))
    logger.info("stage=matched z_rel=%.6g", paired_stats(z)["paired_relative_rms_mean"])

    # Compute VJPs before any surgical in-place parameter ablation changes
    # autograd version counters for the retained baseline graph.
    selected_parameters = {
        "latent_base": model.latent_base,
        "tokenizer_y_weight": model.patch_tokenizer.y_proj.weight,
        "encoder0_cross_out_weight": model.encoder_layers[0].perceiver_block.cross_out_proj.weight,
        "encoder9_cross_out_weight": model.encoder_layers[-1].perceiver_block.cross_out_proj.weight,
    }
    cancellation = gradient_cancellation(W, W_hat, z_flat, selected_parameters, cfg)
    logger.info("stage=vjp_done latent_base_cancel=%.6g", cancellation["latent_base"]["sum_over_sum_norms"])

    interventions: dict[str, dict[str, Any]] = {}
    for name, W_value, X_value, x_mask_value in (
        ("pair_swapped_W_hold_X", W.index_select(0, swap), X, x_mask),
        ("pair_swapped_X_hold_W", W, X.index_select(0, swap), x_mask.index_select(0, swap)),
        ("zero_W_hold_X", torch.zeros_like(W), X, x_mask),
    ):
        _prediction, _debug, active = run_forward(
            model, W_value, X_value, x_mask_value, d_in_mask, d_out_mask, grad=False
        )
        comparison = {
            node: {
                "vs_matched": comparison_stats(matched[node], value),
                **(
                    {"vs_pair_swapped_matched": comparison_stats(matched[node].index_select(0, swap), value)}
                    if name.startswith("pair_swapped")
                    else {}
                ),
                "paired": paired_stats(value),
            }
            for node, value in active.items()
            if node in matched and value.shape == matched[node].shape and int(value.shape[0]) == 18
        }
        interventions[name] = comparison
        logger.info("stage=input_intervention name=%s raw_z_rel=%.6g", name, comparison["encoder.raw_z"]["vs_matched"]["relative_rms"])

    bias_ablations: dict[str, Any] = {}
    ablation_sets = {
        "zero_y_proj_bias": [model.patch_tokenizer.y_proj.bias],
        "zero_all_tokenizer_biases": tokenizer_all_biases(model),
        "zero_all_peri_residual_return_biases": residual_return_biases(model),
    }
    for name, parameters in ablation_sets.items():
        with zero_parameters(parameters):
            _prediction, _debug, active = run_forward(model, W, X, x_mask, d_in_mask, d_out_mask, grad=False)
        bias_ablations[name] = {
            "parameter_count": len(parameters),
            "raw_z": paired_stats(active["encoder.raw_z"]),
            "raw_z_vs_baseline": comparison_stats(matched["encoder.raw_z"], active["encoder.raw_z"]),
            "selected_nodes": {
                node: paired_stats(active[node])
                for node in (
                    "tokenizer.y_proj",
                    "tokenizer.output",
                    "encoder.0.latent_output",
                    "encoder.9.latent_output",
                    "decoder.bridge_output",
                )
            },
        }
        logger.info("stage=bias_ablation name=%s raw_z_pair_rel=%.6g", name, bias_ablations[name]["raw_z"]["paired_relative_rms_mean"])

    base = model.latent_base.unsqueeze(0).expand_as(z)
    batch_mean = z.mean(0, keepdim=True)
    pair_mean = 0.5 * (z + z.index_select(0, swap))
    amplification: dict[str, Any] = {
        "raw_z": paired_stats(z),
        "z_minus_latent_base": paired_stats(z - base),
        "z_minus_batch_mean": paired_stats(z - batch_mean),
        "latent_base": paired_stats(base),
    }
    for family, center in (("latent_base", base), ("batch_mean", batch_mean), ("pair_mean", pair_mean)):
        for factor in (0.0, 1.0, 10.0, 100.0):
            candidate = center + factor * (z - center)
            amplification[f"{family}_factor_{factor:g}"] = {
                "z_stats": paired_stats(candidate),
                "decode": decode_metrics(model, W, debug, candidate, d_in_mask, d_out_mask, cfg),
            }
    logger.info("stage=amplification_done")

    # Seal every core discriminator before the optional fresh-init tail.  This
    # makes a later fresh-baseline failure non-destructive to the main result.
    core_report = {
        "schema": "weightclip_v6_encoder_identity_causal_probe_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "config": str(args.config.resolve()),
        "device": str(device),
        "dtype": "bfloat16 autocast",
        "optimizer_steps": 0,
        "logical_indices": list(batch.logical_indices),
        "baseline_activations": activation_report(matched),
        "input_interventions": interventions,
        "y_projection_contributions": y_projection_contributions(model, W),
        "bias_ablations": bias_ablations,
        "latent_amplification": amplification,
        "gradient_cancellation": cancellation,
    }
    (output_dir / "report.core.json").write_text(
        json.dumps(core_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("stage=core_report_sealed report=%s", output_dir / "report.core.json")

    # Cheap temporal baseline: same resolved config and seed, no checkpoint load.
    fresh_seed = int(cfg.data.seed)
    torch.manual_seed(fresh_seed)
    torch.cuda.manual_seed_all(fresh_seed)
    fresh = build_weight_quantile_vae(build_big_vae_model_config(cfg)).to(device).eval()
    _fresh_prediction, _fresh_debug, fresh_active = run_forward(
        fresh, W, X, x_mask, d_in_mask, d_out_mask, grad=False
    )
    fresh_report: dict[str, Any] = {
        "seed": fresh_seed,
        "note": "same resolved config/seed; model construction baseline, not hash-verified exact training-step-0 state",
        "activations": activation_report(fresh_active),
        "y_projection_contributions": y_projection_contributions(fresh, W),
        "bias_ablations": {},
    }
    for name, parameters in {
        "zero_y_proj_bias": [fresh.patch_tokenizer.y_proj.bias],
        "zero_all_tokenizer_biases": tokenizer_all_biases(fresh),
        "zero_all_peri_residual_return_biases": residual_return_biases(fresh),
    }.items():
        with zero_parameters(parameters):
            _prediction, _debug, active = run_forward(fresh, W, X, x_mask, d_in_mask, d_out_mask, grad=False)
        fresh_report["bias_ablations"][name] = {
            "parameter_count": len(parameters),
            "raw_z": paired_stats(active["encoder.raw_z"]),
            "selected_nodes": {
                node: paired_stats(active[node])
                for node in ("tokenizer.y_proj", "tokenizer.output", "encoder.0.latent_output", "encoder.9.latent_output")
            },
        }
    del fresh
    logger.info("stage=fresh_done")

    report = {
        **core_report,
        "fresh_init": fresh_report,
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("stage=done elapsed=%.1fs report=%s", time.time() - start, output_dir / "report.json")


if __name__ == "__main__":
    main()
