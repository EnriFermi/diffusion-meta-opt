from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
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


def tensor_output(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    return None


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-30) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)


def paired_stats(value: torch.Tensor) -> dict[str, float]:
    original = value.detach().float()
    extra: dict[str, float] = {}
    if original.ndim == 3 and original.shape[1] > 1:
        centered = original - original.mean(dim=1, keepdim=True)
        query_energy = original.square().mean(dim=(1, 2)).clamp_min(1.0e-30)
        centered_energy = centered.square().mean(dim=(1, 2))
        feature_var = centered.square().mean(dim=1)
        diagonal_effective_dimension = feature_var.sum(-1).square() / feature_var.square().sum(-1).clamp_min(1.0e-30)
        # Batched power iteration gives a cheap stable-rank estimate without a
        # 1024x1024 Gram matrix or a full SVD at every decoder layer.
        generator = torch.Generator(device=original.device).manual_seed(84_331)
        vector = torch.randn(
            original.shape[0], original.shape[2], 1,
            device=original.device, dtype=original.dtype, generator=generator,
        )
        vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(1.0e-30)
        for _ in range(6):
            left = centered @ vector
            left = left / left.norm(dim=1, keepdim=True).clamp_min(1.0e-30)
            vector = centered.transpose(1, 2) @ left
            vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(1.0e-30)
        spectral_sq = (centered @ vector).square().sum(dim=(1, 2)).clamp_min(1.0e-30)
        stable_rank = centered.square().sum(dim=(1, 2)) / spectral_sq
        extra = {
            "query_centered_rms": float(centered.square().mean().sqrt().item()),
            "query_centered_energy_fraction_mean": float((centered_energy / query_energy).mean().item()),
            "query_diagonal_effective_dimension_mean": float(diagonal_effective_dimension.mean().item()),
            "query_stable_rank_power_mean": float(stable_rank.mean().item()),
            "query_stable_rank_power_min": float(stable_rank.min().item()),
        }
    value = original.reshape(original.shape[0], -1)
    a, b = value[0::2], value[1::2]
    delta = a - b
    denom = ((a.square().sum(-1) + b.square().sum(-1)) * 0.5).sqrt().clamp_min(1.0e-30)
    common = (a + b) * 0.5
    difference = (a - b) * 0.5
    common_energy = common.square().sum(-1)
    difference_energy = difference.square().sum(-1)
    global_mean = value.mean(0)
    global_energy = value.square().sum(-1).mean().clamp_min(1.0e-30)
    return {
        "rms": float(value.square().mean().sqrt().item()),
        "a_b_delta_rms": float(delta.square().mean().sqrt().item()),
        "a_b_relative_l2_mean": float((delta.norm(dim=-1) / denom).mean().item()),
        "a_b_cosine_mean": float(cosine(a, b).mean().item()),
        "pair_common_fraction_mean": float(
            (common_energy / (common_energy + difference_energy).clamp_min(1.0e-30)).mean().item()
        ),
        "global_common_template_fraction": float(
            (global_mean.square().sum() / global_energy).clamp(0.0, 1.0).item()
        ),
        **extra,
    }


def matched_swapped_stats(matched: torch.Tensor, swapped: torch.Tensor) -> dict[str, float]:
    matched = matched.detach().float().reshape(matched.shape[0], -1)
    swapped = swapped.detach().float().reshape(swapped.shape[0], -1)
    delta = matched - swapped
    denom = matched.norm(dim=-1).clamp_min(1.0e-30)
    return {
        "matched_swapped_delta_rms": float(delta.square().mean().sqrt().item()),
        "matched_swapped_relative_l2_mean": float((delta.norm(dim=-1) / denom).mean().item()),
        "matched_swapped_cosine_mean": float(cosine(matched, swapped).mean().item()),
    }


def gradient_stats(grad: torch.Tensor | None, indices: torch.Tensor) -> dict[str, float]:
    if grad is None:
        return {"grad_rms": 0.0, "grad_l2": 0.0}
    selected = grad.index_select(0, indices).detach().float()
    return {
        "grad_rms": float(selected.square().mean().sqrt().item()),
        "grad_l2": float(selected.norm().item()),
    }


def per_sample_bridge_gradient_report(
    W: torch.Tensor,
    W_hat: torch.Tensor,
    bridge_output: torch.Tensor,
    cfg: Any,
) -> dict[str, float]:
    vectors: list[torch.Tensor] = []
    losses: list[float] = []
    for index in range(int(W.shape[0])):
        loss = direction_loss(W[index : index + 1], W_hat[index : index + 1], cfg)
        grad = torch.autograd.grad(loss, bridge_output, retain_graph=True, allow_unused=False)[0]
        vectors.append(grad[index].detach().float().reshape(-1))
        losses.append(float(loss.detach().item()))
    matrix = torch.stack(vectors)
    norms = matrix.norm(dim=-1)
    mean_gradient = matrix.mean(0)
    normalized = matrix / norms.unsqueeze(-1).clamp_min(1.0e-30)
    pairwise = normalized @ normalized.T
    off_diagonal = pairwise[~torch.eye(len(vectors), device=pairwise.device, dtype=torch.bool)]
    pair_cosines = (normalized[0::2] * normalized[1::2]).sum(-1)
    sum_gradient = matrix.sum(0)
    return {
        "individual_loss_mean": float(sum(losses) / len(losses)),
        "individual_grad_l2_mean": float(norms.mean().item()),
        "individual_grad_l2_min": float(norms.min().item()),
        "individual_grad_l2_max": float(norms.max().item()),
        "mean_grad_l2": float(mean_gradient.norm().item()),
        "sum_grad_l2": float(sum_gradient.norm().item()),
        "cancellation_ratio_sum_over_sum_norms": float((sum_gradient.norm() / norms.sum().clamp_min(1.0e-30)).item()),
        "pairwise_cosine_mean_offdiag": float(off_diagonal.mean().item()),
        "paired_operator_cosine_mean": float(pair_cosines.mean().item()),
        "paired_operator_cosine_min": float(pair_cosines.min().item()),
        "paired_operator_cosine_max": float(pair_cosines.max().item()),
    }


def add_hook(module: torch.nn.Module, name: str, active: dict[str, torch.Tensor], handles: list[Any]) -> None:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        value = tensor_output(output)
        if value is not None and value.ndim >= 1:
            active[name] = value

    handles.append(module.register_forward_hook(hook))


def register_path_hooks(model: torch.nn.Module, active: dict[str, torch.Tensor]) -> list[Any]:
    handles: list[Any] = []
    bridge = model.mandatory_latent_bridge
    for child_name in ("query_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "out_proj"):
        add_hook(getattr(bridge, child_name), f"bridge.{child_name}", active, handles)
    add_hook(bridge, "bridge.output", active, handles)
    for index, layer in enumerate(model.decoder_layers):
        prefix = f"decoder.{index}"
        for child_name in (
            "self_attn_norm",
            "self_q_proj",
            "self_k_proj",
            "self_v_proj",
            "self_q_norm",
            "self_k_norm",
            "self_out_proj",
            "ffn_norm",
        ):
            add_hook(getattr(layer, child_name), f"{prefix}.{child_name}", active, handles)
        add_hook(layer.ffn[0], f"{prefix}.ffn_up", active, handles)
        add_hook(layer.ffn[3], f"{prefix}.ffn_down", active, handles)
        add_hook(layer, f"{prefix}.output", active, handles)
    add_hook(model.q_tokens_norm, "head.q_norm", active, handles)
    add_hook(model.direction_head, "head.direction", active, handles)
    add_hook(model.scale_head, "head.scale", active, handles)
    return handles


def direction_loss(target: torch.Tensor, prediction: torch.Tensor, cfg: Any) -> torch.Tensor:
    total, _details = WeightQuantileVAE.patch_structure_loss(
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
    # The details mapping is telemetry-only and intentionally detached in the
    # production loss API. With every other lambda zero, total is exactly the
    # differentiable directional objective required by the VJP.
    return total


def target_template_oracles(W: torch.Tensor, cfg: Any) -> dict[str, float]:
    patch_size = int(cfg.model.patch_size)
    B, d_in, d_out = W.shape
    if d_in % patch_size:
        raise ValueError("target-template oracle expects complete input-axis patches")
    T = d_in // patch_size
    patches = W.transpose(1, 2).reshape(B, d_out, T, patch_size).float()
    directions = patches / patches.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)

    global_direction = directions.mean(dim=(0, 1, 2))
    global_direction = global_direction / global_direction.norm().clamp_min(1.0e-12)
    global_prediction = global_direction.view(1, 1, 1, patch_size).expand_as(directions)

    per_query_direction = directions.mean(dim=0)
    per_query_direction = per_query_direction / per_query_direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    per_query_prediction = per_query_direction.unsqueeze(0).expand_as(directions)

    per_sample_direction = directions.mean(dim=(1, 2))
    per_sample_direction = per_sample_direction / per_sample_direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    per_sample_prediction = per_sample_direction[:, None, None, :].expand_as(directions)

    def matrix(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(B, d_out, d_in).transpose(1, 2).to(dtype=W.dtype)

    return {
        "one_global_direction_loss": float(direction_loss(W, matrix(global_prediction), cfg).detach().item()),
        "per_query_position_shared_across_samples_loss": float(
            direction_loss(W, matrix(per_query_prediction), cfg).detach().item()
        ),
        "per_sample_global_direction_loss": float(direction_loss(W, matrix(per_sample_prediction), cfg).detach().item()),
        "identity_direction_loss": float(direction_loss(W, W, cfg).detach().item()),
    }


def rowspace_report(model: torch.nn.Module, z: torch.Tensor) -> dict[str, Any]:
    bridge = model.mandatory_latent_bridge
    wk = bridge.k_proj.weight.detach().float()
    wv = bridge.v_proj.weight.detach().float()
    stacked = torch.cat((wk, wv), dim=0)
    _u, singular, vh = torch.linalg.svd(stacked, full_matrices=False)
    dz = (z[0::2] - z[1::2]).detach().float().view(-1, z.shape[-1])
    output: dict[str, Any] = {
        "stacked_shape": list(stacked.shape),
        "sigma_max": float(singular.max().item()),
        "sigma_min": float(singular.min().item()),
        "condition_number": float((singular.max() / singular.min().clamp_min(1.0e-30)).item()),
    }
    for relative_threshold in (1.0e-3, 1.0e-4, 1.0e-6):
        rank = int((singular > singular.max() * relative_threshold).sum().item())
        basis = vh[:rank]
        projected = (dz @ basis.T) @ basis
        total_energy = dz.square().sum().clamp_min(1.0e-30)
        row_energy = projected.square().sum()
        output[f"rank_rel_{relative_threshold:g}"] = rank
        output[f"dz_rowspace_fraction_rel_{relative_threshold:g}"] = float((row_energy / total_energy).item())
        output[f"dz_nullspace_fraction_rel_{relative_threshold:g}"] = float(
            ((dz - projected).square().sum() / total_energy).item()
        )
    dz_norm = dz.norm(dim=-1).clamp_min(1.0e-30)
    output["wk_dz_gain_mean"] = float(((dz @ wk.T).norm(dim=-1) / dz_norm).mean().item())
    output["wv_dz_gain_mean"] = float(((dz @ wv.T).norm(dim=-1) / dz_norm).mean().item())
    output["stacked_dz_gain_mean"] = float(((dz @ stacked.T).norm(dim=-1) / dz_norm).mean().item())
    return output


def attention_report(acts: dict[str, torch.Tensor], heads: int, head_dim: int) -> tuple[dict[str, float], torch.Tensor]:
    q = acts["bridge.q_norm"].detach().float()
    k = acts["bridge.k_norm"].detach().float()
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(head_dim))
    probs = logits.softmax(-1)
    entropy = -(probs * probs.clamp_min(1.0e-30).log()).sum(-1) / math.log(float(probs.shape[-1]))
    report = {
        **paired_stats(probs),
        "entropy_normalized_mean": float(entropy.mean().item()),
        "max_probability_mean": float(probs.max(-1).values.mean().item()),
        "shape_BHQK": list(probs.shape),
        "heads": int(heads),
        "head_dim": int(head_dim),
    }
    return report, probs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "progress.log")],
    )
    logger = logging.getLogger("v5-two-op-identity-path")
    start = time.time()
    logger.info("stage=start checkpoint=%s config=%s device=%s dtype=bf16 seed=fixed", args.checkpoint, args.config, args.device)
    cfg = OmegaConf.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model_state"]
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device(args.device)
    model.to(device).eval()
    logger.info("stage=model_ready step=%s params=%s", checkpoint_step, sum(p.numel() for p in model.parameters()))

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
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
            collate_fn=_identity_sample_collate,
        )
        mixer = BalancedOperatorBankMixer(dataset, _flatten_loader_batches(iter(loader)), start_index=0)
        batch = _fetch_presliced_training_batch_cpu(dataset_iter=mixer, batch_size=18, logger=logger)
    if tuple(batch.logical_indices) != tuple(range(18)):
        raise RuntimeError(f"unexpected logical identities: {batch.logical_indices}")
    W = batch.W.to(device)
    X = batch.x.to(device)
    x_mask = batch.x_mask.to(device)
    d_in_mask = batch.d_in_mask.to(device)
    d_out_mask = batch.d_out_mask.to(device)
    idx_a = torch.arange(0, 18, 2, device=device)
    idx_b = torch.arange(1, 18, 2, device=device)
    swap = torch.arange(18, device=device) ^ 1
    logger.info("stage=data_ready W=%s X=%s logical=%s", tuple(W.shape), tuple(X.shape), batch.logical_indices)
    template_oracles = target_template_oracles(W, cfg)
    logger.info("stage=target_oracles %s", json.dumps(template_oracles, sort_keys=True))

    matched_acts: dict[str, torch.Tensor] = {}
    handles = register_path_hooks(model, matched_acts)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        W_hat, _mu, _logvar, pred_dirs, debug = model.forward_debug(
            W, X, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
        )
        matched_acts["encoder.raw_decoder_z"] = debug["latent_decoder_z"]
        matched_acts["decoder.query_input"] = debug["decoder_queries_init"]
        matched_acts["output.pred_dirs"] = pred_dirs
        matched_acts["output.W_hat"] = W_hat
        loss_a = direction_loss(W.index_select(0, idx_a), W_hat.index_select(0, idx_a), cfg)
        loss_b = direction_loss(W.index_select(0, idx_b), W_hat.index_select(0, idx_b), cfg)
    logger.info("stage=matched_forward loss_a=%.9f loss_b=%.9f nodes=%s", loss_a.item(), loss_b.item(), len(matched_acts))

    grad_nodes = {
        name: value
        for name, value in matched_acts.items()
        if value.requires_grad and value.ndim > 0 and int(value.shape[0]) == 18
    }
    names = list(grad_nodes)
    values = [grad_nodes[name] for name in names]
    grads_a = torch.autograd.grad(loss_a, values, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, values, retain_graph=True, allow_unused=True)
    baseline_z_grad = torch.autograd.grad(
        (loss_a + loss_b) * 0.5,
        debug["latent_decoder_z"],
        retain_graph=True,
        allow_unused=False,
    )[0]
    grad_rows: list[dict[str, Any]] = []
    for name, ga, gb in zip(names, grads_a, grads_b):
        grad_rows.append(
            {
                "node": name,
                "operator_a_loss": float(loss_a.detach().item()),
                "operator_b_loss": float(loss_b.detach().item()),
                **{f"operator_a_{key}": value for key, value in gradient_stats(ga, idx_a).items()},
                **{f"operator_b_{key}": value for key, value in gradient_stats(gb, idx_b).items()},
            }
        )
    logger.info("stage=vjp_done nodes=%s", len(grad_rows))

    bridge_sample_grads = per_sample_bridge_gradient_report(
        W,
        W_hat,
        matched_acts["bridge.output"],
        cfg,
    )
    logger.info("stage=per_sample_bridge_vjp cancellation=%.6g", bridge_sample_grads["cancellation_ratio_sum_over_sum_norms"])

    activation_rows = [{"node": name, **paired_stats(value)} for name, value in matched_acts.items() if value.shape[0] == 18]
    rowspace = rowspace_report(model, debug["latent_decoder_z"].view(18, int(cfg.model.big_vae.num_latents), int(cfg.model.big_vae.d_lat)))
    attention_matched, probs_matched = attention_report(
        matched_acts,
        heads=int(cfg.model.big_vae.decoder_bridge_n_heads),
        head_dim=int(cfg.model.big_vae.decoder_bridge_attn_dim) // int(cfg.model.big_vae.decoder_bridge_n_heads),
    )

    swapped_acts: dict[str, torch.Tensor] = {}
    for handle in handles:
        handle.remove()
    handles = register_path_hooks(model, swapped_acts)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        swapped_W_hat, _z, _zero_logvar, swapped_dirs = model._decode_from_decoder_latent(
            debug["latent_decoder_z"].index_select(0, swap),
            dist_patch_by_patch=debug["dist_patch_by_patch"],
            patch_mask=debug["patch_mask"],
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            d_in=int(W.shape[1]),
            d_out=int(W.shape[2]),
            d_in_pad=int(debug["d_in_pad"]),
            T=int(debug["T"]),
        )
        swapped_acts["output.pred_dirs"] = swapped_dirs
        swapped_acts["output.W_hat"] = swapped_W_hat
    for handle in handles:
        handle.remove()
    matched_swapped_rows = [
        {"node": name, **matched_swapped_stats(matched_acts[name], swapped_acts[name])}
        for name in sorted(set(matched_acts) & set(swapped_acts))
        if matched_acts[name].shape == swapped_acts[name].shape and matched_acts[name].shape[0] == 18
    ]
    attention_swapped, probs_swapped = attention_report(
        swapped_acts,
        heads=int(cfg.model.big_vae.decoder_bridge_n_heads),
        head_dim=int(cfg.model.big_vae.decoder_bridge_attn_dim) // int(cfg.model.big_vae.decoder_bridge_n_heads),
    )
    attention_swap_delta = matched_swapped_stats(probs_matched, probs_swapped)
    logger.info("stage=swapped_forward nodes=%s", len(swapped_acts))

    # Frozen, zero-step intervention: restore the already-computed absolute
    # query positional carrier immediately after the mandatory bridge.
    intervention_acts: dict[str, torch.Tensor] = {}
    handles = register_path_hooks(model, intervention_acts)
    q_pos_emb = debug["decoder_query_pos_emb"].detach()

    def inject_qpos(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        result = output + q_pos_emb.to(device=output.device, dtype=output.dtype)
        intervention_acts["intervention.bridge_plus_qpos"] = result
        return result

    handles.append(model.mandatory_latent_bridge.register_forward_hook(inject_qpos))
    intervention_z = debug["latent_decoder_z"].detach().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        intervention_W_hat, _z, _zero_logvar, intervention_dirs = model._decode_from_decoder_latent(
            intervention_z,
            dist_patch_by_patch=debug["dist_patch_by_patch"].detach(),
            patch_mask=debug["patch_mask"],
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            d_in=int(W.shape[1]),
            d_out=int(W.shape[2]),
            d_in_pad=int(debug["d_in_pad"]),
            T=int(debug["T"]),
        )
        intervention_acts["output.pred_dirs"] = intervention_dirs
        intervention_acts["output.W_hat"] = intervention_W_hat
        intervention_loss_a = direction_loss(W.index_select(0, idx_a), intervention_W_hat.index_select(0, idx_a), cfg)
        intervention_loss_b = direction_loss(W.index_select(0, idx_b), intervention_W_hat.index_select(0, idx_b), cfg)
    intervention_z_grad = torch.autograd.grad(
        (intervention_loss_a + intervention_loss_b) * 0.5,
        intervention_z,
        retain_graph=False,
        allow_unused=False,
    )[0]
    for handle in handles:
        handle.remove()
    intervention_rows = [
        {"node": name, **paired_stats(value)}
        for name, value in intervention_acts.items()
        if value.shape[0] == 18
    ]
    logger.info(
        "stage=qpos_intervention loss_a=%.9f loss_b=%.9f z_grad_rms=%.6g",
        intervention_loss_a.item(), intervention_loss_b.item(), intervention_z_grad.float().square().mean().sqrt().item(),
    )

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in rows for key in row})
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("activation_deltas.csv", activation_rows)
    write_csv("directional_vjp.csv", grad_rows)
    write_csv("matched_swapped_deltas.csv", matched_swapped_rows)
    write_csv("qpos_intervention_activations.csv", intervention_rows)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "config": str(args.config.resolve()),
        "device": str(device),
        "dtype": "bfloat16 autocast",
        "optimizer_steps": 0,
        "logical_indices": list(batch.logical_indices),
        "operator_a_direction_loss": float(loss_a.detach().item()),
        "operator_b_direction_loss": float(loss_b.detach().item()),
        "target_template_oracles": template_oracles,
        "rowspace": rowspace,
        "bridge_attention_matched": attention_matched,
        "bridge_attention_swapped": attention_swapped,
        "bridge_attention_matched_swapped": attention_swap_delta,
        "per_sample_bridge_directional_gradients": bridge_sample_grads,
        "qpos_intervention": {
            "definition": "q_hidden := mandatory_bridge_output + decoder_query_pos_emb; frozen weights; zero optimizer steps",
            "baseline_operator_a_direction_loss": float(loss_a.detach().item()),
            "baseline_operator_b_direction_loss": float(loss_b.detach().item()),
            "intervention_operator_a_direction_loss": float(intervention_loss_a.detach().item()),
            "intervention_operator_b_direction_loss": float(intervention_loss_b.detach().item()),
            "baseline_z_grad_rms": float(baseline_z_grad.detach().float().square().mean().sqrt().item()),
            "baseline_z_grad_l2": float(baseline_z_grad.detach().float().norm().item()),
            "intervention_z_grad_rms": float(intervention_z_grad.detach().float().square().mean().sqrt().item()),
            "intervention_z_grad_l2": float(intervention_z_grad.detach().float().norm().item()),
        },
        "elapsed_seconds": time.time() - start,
        "artifacts": {
            "activation_deltas": str(output_dir / "activation_deltas.csv"),
            "directional_vjp": str(output_dir / "directional_vjp.csv"),
            "matched_swapped_deltas": str(output_dir / "matched_swapped_deltas.csv"),
            "qpos_intervention_activations": str(output_dir / "qpos_intervention_activations.csv"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("stage=done elapsed=%.1fs report=%s", time.time() - start, output_dir / "report.json")


if __name__ == "__main__":
    main()
