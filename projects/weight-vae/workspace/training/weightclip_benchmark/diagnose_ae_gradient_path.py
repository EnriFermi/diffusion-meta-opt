from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.big_vae.runtime import _compute_latent_sampling_gate_for_step


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace AE gradients from loss through decoder into encoder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument(
        "--interventions-only",
        action="store_true",
        help="Run decoder latent/context interventions without boundary or parameter backward passes.",
    )
    parser.add_argument(
        "--postnorm-modes",
        default="trained",
        help="Comma-separated: trained,bypass_decoder,bypass_encoder,bypass_all",
    )
    return parser.parse_args()


def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    value = tensor.detach()
    finite = torch.isfinite(value)
    safe = torch.where(finite, value, torch.zeros_like(value)).float()
    count = max(1, int(value.numel()))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "rms": float(torch.linalg.vector_norm(safe).item() / math.sqrt(count)),
        "l2": float(torch.linalg.vector_norm(safe).item()),
        "max_abs": float(safe.abs().max().item()) if value.numel() else 0.0,
        "zero_fraction": float((safe == 0).float().mean().item()) if value.numel() else 1.0,
        "finite_fraction": float(finite.float().mean().item()) if value.numel() else 1.0,
    }


def _tensor_items(value: Any) -> list[tuple[int, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(0, value)]
    if isinstance(value, (tuple, list)):
        return [(idx, item) for idx, item in enumerate(value) if torch.is_tensor(item)]
    return []


class GradientPathTrace:
    def __init__(self, modules: Iterable[tuple[str, torch.nn.Module]]) -> None:
        self.rows: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        for name, module in modules:
            self._handles.append(module.register_forward_hook(self._forward_hook(name)))
            self._handles.append(module.register_full_backward_hook(self._backward_hook(name)))

    def clear(self) -> None:
        self.rows.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _forward_hook(self, name: str):
        def hook(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
            for side, value in (("input", inputs), ("output", output)):
                for tensor_index, tensor in _tensor_items(value):
                    stats = _tensor_stats(tensor)
                    if stats is not None:
                        self.rows.append(
                            {"phase": "activation", "module": name, "side": side, "tensor_index": tensor_index, **stats}
                        )

        return hook

    def _backward_hook(self, name: str):
        def hook(_module: torch.nn.Module, grad_input: Any, grad_output: Any) -> None:
            for side, value in (("grad_input", grad_input), ("grad_output", grad_output)):
                for tensor_index, tensor in _tensor_items(value):
                    stats = _tensor_stats(tensor)
                    if stats is not None:
                        self.rows.append(
                            {"phase": "gradient", "module": name, "side": side, "tensor_index": tensor_index, **stats}
                        )

        return hook


def _trace_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    modules: list[tuple[str, torch.nn.Module]] = [
        ("patch_tokenizer", model.patch_tokenizer),
        ("patch_token_proj", model.patch_token_proj),
        ("latent_norm", model.latent_norm),
        ("latent_to_decoder", model.latent_to_decoder),
        ("q_tokens_norm", model.q_tokens_norm),
        ("direction_head", model.direction_head),
        ("scale_head", model.scale_head),
    ]
    patch_blocks = getattr(model.patch_tokenizer, "blocks", None)
    if patch_blocks is not None:
        for idx, block in enumerate(patch_blocks):
            modules.extend(
                [
                    (f"patch_tokenizer.{idx}.base_return", block.base_fc2),
                    (f"patch_tokenizer.{idx}.conditioning_return", block.cond_fc2),
                    (f"patch_tokenizer.{idx}.gate", block.gate_fc),
                ]
            )
    for idx, layer in enumerate(model.encoder_layers):
        modules.extend(
            [
                (f"encoder.{idx}.local", layer.local_block),
                (f"encoder.{idx}.local.attn_return", layer.local_block.out_proj),
                (f"encoder.{idx}.local.ffn_return", layer.local_block.ffn.fc2),
                (f"encoder.{idx}.perceiver", layer.perceiver_block),
                (f"encoder.{idx}.perceiver.cross_return", layer.perceiver_block.cross_out_proj),
                (f"encoder.{idx}.perceiver.self_return", layer.perceiver_block.self_out_proj),
                (f"encoder.{idx}.perceiver.ffn_return", layer.perceiver_block.ffn[-2]),
                (f"encoder.{idx}.whole", layer),
            ]
        )
    adapters = getattr(model, "encoder_conditioning_adapters", None)
    if adapters is not None:
        for idx, module in enumerate(adapters):
            modules.extend(
                [
                    (f"encoder.{idx}.conditioning", module),
                    (f"encoder.{idx}.conditioning.return", module.mix_out),
                    (f"encoder.{idx}.conditioning.gate", module.gate_out),
                ]
            )
    weight_post_norms = getattr(model, "encoder_weight_post_norms", None)
    if weight_post_norms is not None:
        modules.extend((f"encoder.{idx}.weight_postnorm", module) for idx, module in enumerate(weight_post_norms))
    latent_post_norms = getattr(model, "encoder_latent_post_norms", None)
    if latent_post_norms is not None:
        modules.extend((f"encoder.{idx}.latent_postnorm", module) for idx, module in enumerate(latent_post_norms))
    feedback = getattr(model, "latent_to_weight_feedback", None)
    if feedback is not None:
        modules.extend(
            [
                ("encoder.latent_to_weight_feedback", feedback),
                ("encoder.latent_to_weight_feedback.return", feedback.out),
            ]
        )
    for idx, layer in enumerate(model.decoder_layers):
        modules.extend(
            [
                (f"decoder.{idx}.whole", layer),
                (f"decoder.{idx}.cross_return", layer.out_proj),
                (f"decoder.{idx}.self_return", layer.self_out_proj),
                (f"decoder.{idx}.ffn_return", layer.ffn[-2]),
            ]
        )
    decoder_post_norms = getattr(model, "decoder_post_norms", None)
    if decoder_post_norms is not None:
        modules.extend((f"decoder.{idx}.postnorm", module) for idx, module in enumerate(decoder_post_norms))
    return modules


def _parse_postnorm_modes(raw: str) -> list[str]:
    allowed = {"trained", "bypass_decoder", "bypass_encoder", "bypass_all"}
    modes = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("postnorm modes must be a non-empty unique comma-separated list")
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unsupported postnorm modes: {unknown}")
    return modes


def _set_postnorm_mode(
    model: torch.nn.Module,
    mode: str,
    *,
    encoder_weight_post_norms: torch.nn.ModuleList | None,
    encoder_latent_post_norms: torch.nn.ModuleList | None,
    decoder_post_norms: torch.nn.ModuleList | None,
) -> None:
    model.encoder_weight_post_norms = None if mode in {"bypass_encoder", "bypass_all"} else encoder_weight_post_norms
    model.encoder_latent_post_norms = None if mode in {"bypass_encoder", "bypass_all"} else encoder_latent_post_norms
    model.decoder_post_norms = None if mode in {"bypass_decoder", "bypass_all"} else decoder_post_norms


def _losses(
    model: torch.nn.Module,
    cfg: Any,
    W: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    W_hat, mu, logvar, pred_dirs, debug_info = model.forward_debug(
        W,
        x,
        x_mask=x_mask,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    behavioral_cfg = cfg.train.behavioral_loss
    struct_cfg = cfg.train.struct_loss
    operator = WeightQuantileVAE.operator_recon_loss(
        x, W, W_hat, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
    )
    behavioral_dir, behavioral_scale = WeightQuantileVAE.operator_direction_scale_loss(
        x,
        W,
        W_hat,
        x_mask=x_mask,
        d_out_mask=d_out_mask,
        gamma=float(behavioral_cfg.gamma),
        huber_delta=float(behavioral_cfg.huber_delta),
    )
    behavioral = (
        float(behavioral_cfg.lambda_operator) * operator
        + float(behavioral_cfg.lambda_dir) * behavioral_dir
        + float(behavioral_cfg.lambda_scale) * behavioral_scale
    )
    structural, struct_details = WeightQuantileVAE.patch_structure_loss(
        W,
        W_hat,
        patch_size=int(cfg.model.patch_size),
        gamma=float(struct_cfg.gamma),
        lambda_dir=float(struct_cfg.lambda_dir),
        lambda_scale=float(struct_cfg.lambda_scale),
        lambda_rec=float(struct_cfg.lambda_rec),
        lambda_rel=float(struct_cfg.lambda_rel),
        huber_delta=float(struct_cfg.huber_delta),
        pred_dirs=pred_dirs,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
    )
    reconstruction = float(cfg.train.behavioral_coef) * behavioral + float(cfg.train.structural_coef) * structural
    kl = model.latent_kl_loss(mu, logvar)
    kl_beta = float(cfg.train.kl_beta)
    total = reconstruction + kl_beta * kl
    details = {
        "total": total,
        "reconstruction": reconstruction,
        "behavioral": behavioral,
        "behavioral_operator": operator,
        "behavioral_direction": behavioral_dir,
        "behavioral_scale": behavioral_scale,
        "structural": structural,
        "structural_direction": struct_details["L_dir"],
        "structural_scale": struct_details["L_scale"],
        "kl": kl,
        "weighted_kl": kl_beta * kl,
    }
    return total, details, (W_hat, mu, logvar, pred_dirs, debug_info)


def _latent_intervention_losses(
    model: torch.nn.Module,
    cfg: Any,
    *,
    W: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
    debug_info: dict[str, Any],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    canonical_W_hat: torch.Tensor,
    canonical_pred_dirs: torch.Tensor,
) -> dict[str, float]:
    """Measure whether the trained decoder uses the sample-specific latent."""
    decoder_z = debug_info["latent_decoder_z"].detach()
    mu = mu.detach()
    posterior_std = torch.exp(0.5 * logvar.detach())
    gate = float(debug_info["latent_sampling_gate"].detach().float().item())
    noise_delta = decoder_z - mu
    matched_context = debug_info["dist_patch_by_patch"]
    shuffled_z = decoder_z.roll(1, dims=0)
    zero_z = torch.zeros_like(decoder_z)
    mean_z = decoder_z.mean(dim=0, keepdim=True).expand_as(decoder_z)
    shuffled_context = matched_context.roll(1, dims=0)
    zero_context = torch.zeros_like(matched_context)
    variants = {
        "sampled_matched": (decoder_z, matched_context),
        "sampled_full_shuffle": (shuffled_z, matched_context),
        "same_noise_shuffled_mu": (mu.roll(1, dims=0) + noise_delta, matched_context),
        "same_noise_zero_mu": (noise_delta, matched_context),
        "mu_only_matched": (mu, matched_context),
        "mu_only_shuffled": (mu.roll(1, dims=0), matched_context),
        "zero": (zero_z, matched_context),
        "shuffled_context": (decoder_z, shuffled_context),
        "zero_context": (decoder_z, zero_context),
        # Full z x Xctx factorial.  The paired-shuffle cell keeps the donor
        # latent and context coherent, while the crossed cells separate
        # redundancy/competition from a decoder interface that ignores z.
        "shuffled_z_shuffled_context": (shuffled_z, shuffled_context),
        "shuffled_z_zero_context": (shuffled_z, zero_context),
        "zero_z_shuffled_context": (zero_z, shuffled_context),
        "zero_z_zero_context": (zero_z, zero_context),
        # Preserve the batch-common latent while varying only the centered,
        # sample-specific component.  This detects a useful signal that is
        # present in z but drowned by a common cross-attention direction.
        "mean_z": (mean_z, matched_context),
        "centered_z_x2": (mean_z + 2.0 * (decoder_z - mean_z), matched_context),
        "centered_z_x4": (mean_z + 4.0 * (decoder_z - mean_z), matched_context),
    }
    losses: dict[str, float] = {}
    active_capture: dict[str, list[torch.Tensor]] = {"raw": [], "normalized": []}
    capture_handles = []
    for decoder_layer in model.decoder_layers:
        capture_handles.append(
            decoder_layer.out_proj.register_forward_hook(
                lambda _module, _inputs, output: active_capture["raw"].append(output.detach())
            )
        )
        capture_handles.append(
            decoder_layer.cross_return_norm.register_forward_hook(
                lambda _module, _inputs, output: active_capture["normalized"].append(output.detach())
            )
        )
    matched_captures: dict[str, list[torch.Tensor]] | None = None
    try:
        for name, (candidate_z, candidate_context) in variants.items():
            active_capture["raw"] = []
            active_capture["normalized"] = []
            decoded = model._decode_from_decoder_latent(
                candidate_z,
                dist_patch_by_patch=candidate_context,
                patch_mask=debug_info["patch_mask"],
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
                d_in=int(W.shape[1]),
                d_out=int(W.shape[2]),
                d_in_pad=int(debug_info["d_in_pad"]),
                T=int(debug_info["T"]),
            )
            W_hat, _z, _logvar, pred_dirs = decoded
            if len(active_capture["raw"]) != len(model.decoder_layers) or len(active_capture["normalized"]) != len(
                model.decoder_layers
            ):
                raise RuntimeError("decoder cross-return capture count does not match decoder depth")
            if name == "sampled_matched":
                matched_captures = {
                    key: [value.detach().clone() for value in values]
                    for key, values in active_capture.items()
                }
            if matched_captures is None:
                raise RuntimeError("sampled_matched must be the first latent intervention")
            for capture_kind in ("raw", "normalized"):
                for layer_idx, (candidate_value, matched_value) in enumerate(
                    zip(active_capture[capture_kind], matched_captures[capture_kind], strict=True)
                ):
                    candidate_float = candidate_value.float()
                    delta = candidate_float - matched_value.float()
                    losses[f"{name}__decoder_{layer_idx}_cross_{capture_kind}_rms"] = float(
                        candidate_float.square().mean().sqrt().item()
                    )
                    losses[f"{name}__decoder_{layer_idx}_cross_{capture_kind}_delta_rms"] = float(
                        delta.square().mean().sqrt().item()
                    )
                    centered = candidate_float - candidate_float.mean(dim=0, keepdim=True)
                    losses[f"{name}__decoder_{layer_idx}_cross_{capture_kind}_batch_centered_rms"] = float(
                        centered.square().mean().sqrt().item()
                    )
            if name == "sampled_matched":
                w_max_abs = float((W_hat - canonical_W_hat).detach().float().abs().max().item())
                dirs_max_abs = float((pred_dirs - canonical_pred_dirs).detach().float().abs().max().item())
                if w_max_abs > 1e-6 or dirs_max_abs > 1e-6:
                    raise RuntimeError(
                        "latent intervention decoder does not reproduce canonical forward: "
                        f"W_hat max_abs={w_max_abs}, pred_dirs max_abs={dirs_max_abs}"
                    )
                losses["canonical_replay_W_hat_max_abs"] = w_max_abs
                losses["canonical_replay_pred_dirs_max_abs"] = dirs_max_abs
            operator = WeightQuantileVAE.operator_recon_loss(
                x, W, W_hat, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
            )
            behavioral_cfg = cfg.train.behavioral_loss
            behavioral_dir, behavioral_scale = WeightQuantileVAE.operator_direction_scale_loss(
                x,
                W,
                W_hat,
                x_mask=x_mask,
                d_out_mask=d_out_mask,
                gamma=float(behavioral_cfg.gamma),
                huber_delta=float(behavioral_cfg.huber_delta),
            )
            behavioral = (
                float(behavioral_cfg.lambda_operator) * operator
                + float(behavioral_cfg.lambda_dir) * behavioral_dir
                + float(behavioral_cfg.lambda_scale) * behavioral_scale
            )
            structural, _ = WeightQuantileVAE.patch_structure_loss(
                W,
                W_hat,
                patch_size=int(cfg.model.patch_size),
                gamma=float(cfg.train.struct_loss.gamma),
                lambda_dir=float(cfg.train.struct_loss.lambda_dir),
                lambda_scale=float(cfg.train.struct_loss.lambda_scale),
                lambda_rec=float(cfg.train.struct_loss.lambda_rec),
                lambda_rel=float(cfg.train.struct_loss.lambda_rel),
                huber_delta=float(cfg.train.struct_loss.huber_delta),
                pred_dirs=pred_dirs,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
            )
            reconstruction = (
                float(cfg.train.behavioral_coef) * behavioral
                + float(cfg.train.structural_coef) * structural
            )
            losses[name] = float(reconstruction.detach().item())
            component_values = {
                "behavioral_operator": operator,
                "behavioral_direction": behavioral_dir,
                "behavioral_scale": behavioral_scale,
                "behavioral": behavioral,
                "structural": structural,
            }
            for component_name, component_value in component_values.items():
                losses[f"{name}__{component_name}"] = float(component_value.detach().item())

            valid_weight_mask = (
                d_in_mask.to(dtype=torch.bool).unsqueeze(2)
                & d_out_mask.to(dtype=torch.bool).unsqueeze(1)
            )
            weight_delta = (W_hat - canonical_W_hat).detach().float()
            valid_weight_delta = weight_delta.masked_select(valid_weight_mask)
            losses[f"{name}__W_hat_delta_rms"] = (
                float(valid_weight_delta.square().mean().sqrt().item()) if valid_weight_delta.numel() else 0.0
            )
            losses[f"{name}__W_hat_delta_max_abs"] = (
                float(valid_weight_delta.abs().max().item()) if valid_weight_delta.numel() else 0.0
            )
            pred_delta = (pred_dirs - canonical_pred_dirs).detach().float()
            losses[f"{name}__pred_dirs_delta_rms"] = float(pred_delta.square().mean().sqrt().item())
    finally:
        for handle in capture_handles:
            handle.remove()
    sampled_matched = losses["sampled_matched"]
    for name in ("sampled_full_shuffle", "same_noise_shuffled_mu", "same_noise_zero_mu", "zero"):
        losses[f"{name}_minus_sampled_matched"] = losses[name] - sampled_matched
    losses["mu_only_shuffled_minus_matched"] = losses["mu_only_shuffled"] - losses["mu_only_matched"]
    losses["shuffled_context_minus_matched"] = losses["shuffled_context"] - sampled_matched
    losses["zero_context_minus_matched"] = losses["zero_context"] - sampled_matched
    losses["shuffle_interaction"] = (
        losses["shuffled_z_shuffled_context"]
        - losses["sampled_full_shuffle"]
        - losses["shuffled_context"]
        + sampled_matched
    )
    losses["zero_interaction"] = (
        losses["zero_z_zero_context"]
        - losses["zero"]
        - losses["zero_context"]
        + sampled_matched
    )
    losses["sampling_gate"] = gate
    losses["posterior_std_mean"] = float(posterior_std.float().mean().item())
    return losses


def _boundary_component_gradients(
    *,
    reconstruction: torch.Tensor,
    weighted_kl: torch.Tensor,
    mu: torch.Tensor,
    debug_info: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    targets = {
        "decoder_z": debug_info["latent_decoder_z"],
        "mu": mu,
        "raw_encoder_latent": debug_info["latent_base_z"],
    }
    # Ask for each intermediate separately. Requesting decoder_z and its
    # ancestors in one autograd.grad call stops the reported derivative at the
    # decoder_z target and incorrectly marks the ancestor gradients unused.
    def gradient_or_none(loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        # A deterministic AE has an exact constant-zero KL term.  Treat that
        # as a zero/absent gradient instead of asking autograd to differentiate
        # a tensor with requires_grad=False.
        if not loss.requires_grad:
            return None
        return torch.autograd.grad(loss, target, retain_graph=True, allow_unused=True)[0]

    rec_grads = tuple(gradient_or_none(reconstruction, target) for target in targets.values())
    kl_grads = tuple(gradient_or_none(weighted_kl, target) for target in targets.values())
    output: dict[str, dict[str, float | None]] = {}
    for (name, _target), rec_grad, kl_grad in zip(targets.items(), rec_grads, kl_grads, strict=True):
        rec_rms = _tensor_stats(rec_grad)
        kl_rms = _tensor_stats(kl_grad)
        cosine: float | None = None
        if rec_grad is not None and kl_grad is not None:
            rec_flat = rec_grad.detach().float().reshape(-1)
            kl_flat = kl_grad.detach().float().reshape(-1)
            denom = torch.linalg.vector_norm(rec_flat) * torch.linalg.vector_norm(kl_flat)
            if float(denom.item()) > 0.0:
                cosine = float(torch.dot(rec_flat, kl_flat).div(denom).item())
        output[name] = {
            "reconstruction_grad_rms": None if rec_rms is None else float(rec_rms["rms"]),
            "weighted_kl_grad_rms": None if kl_rms is None else float(kl_rms["rms"]),
            "weighted_kl_to_reconstruction_rms_ratio": (
                None
                if rec_rms is None or kl_rms is None or float(rec_rms["rms"]) == 0.0
                else float(kl_rms["rms"]) / float(rec_rms["rms"])
            ),
            "gradient_cosine": cosine,
        }
    return output


def _parameter_gradient_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        stats = _tensor_stats(parameter.grad)
        if stats is not None:
            rows.append({"phase": "parameter_gradient", "module": name, "side": "parameter", "tensor_index": 0, **stats})
    return rows


def _aggregate_parameter_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefixes = [
        "patch_tokenizer",
        "patch_token_proj",
        "distribution_encoder",
        "latent_norm",
        "latent_to_decoder",
        "direction_head",
        "scale_head",
    ]
    prefixes += [f"encoder_layers.{idx}.local_block" for idx in range(10)]
    prefixes += [f"encoder_layers.{idx}.perceiver_block" for idx in range(10)]
    prefixes += [f"encoder_conditioning_adapters.{idx}" for idx in range(10)]
    prefixes += [f"decoder_layers.{idx}" for idx in range(8)]
    output: list[dict[str, Any]] = []
    for prefix in prefixes:
        selected = [row for row in rows if str(row["module"]).startswith(prefix)]
        if not selected:
            continue
        total_sq = sum(float(row["l2"]) ** 2 for row in selected)
        total_numel = sum(int(row["numel"]) for row in selected)
        output.append(
            {
                "module": prefix,
                "parameter_tensors": len(selected),
                "numel": total_numel,
                "grad_l2": math.sqrt(total_sq),
                "grad_rms": math.sqrt(total_sq / max(1, total_numel)),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / f"gradient-path-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("ae-gradient-path")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_step = int(checkpoint["step"])
    cfg = OmegaConf.create(checkpoint["config"])
    model_state = checkpoint["model_state"]
    del checkpoint
    device = torch.device(args.device)
    model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
    model.load_state_dict(model_state, strict=True)
    del model_state
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    gate_cfg = cfg.train.get("latent_sampling_gate", {})
    sampling_gate = _compute_latent_sampling_gate_for_step(
        checkpoint_step,
        schedule_enabled=bool(gate_cfg.get("enabled", True)),
        start_step=int(gate_cfg.get("start_step", 0)),
        ramp_steps=int(gate_cfg.get("ramp_steps", 0)),
        start_value=float(gate_cfg.get("start_value", 0.0)),
        end_value=float(gate_cfg.get("end_value", 1.0)),
    )
    model.set_latent_sampling_gate(sampling_gate)
    logger.info(
        "Model ready: checkpoint_step=%s device=%s batches=%s batch_size=%s postnorm_modes=%s sampling_gate=%.6f",
        checkpoint_step,
        device,
        args.batches,
        args.batch_size,
        args.postnorm_modes,
        sampling_gate,
    )

    pair = str(cfg.train.operator_bank.pair_manifest)
    seed = int(cfg.data.seed)
    torch.manual_seed(seed + checkpoint_step)
    start_index = (
        int(args.start_index)
        if args.start_index is not None
        else checkpoint_step * int(cfg.train.slice_batch_size) * int(cfg.train.grad_accum_steps)
    )
    postnorm_modes = _parse_postnorm_modes(args.postnorm_modes)
    trace = GradientPathTrace(_trace_modules(model))
    all_trace_rows: list[dict[str, Any]] = []
    all_parameter_groups: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    cached_batches = []
    try:
        with operator_bank_data_pipeline(
            pair,
            seed=seed,
            repeat=True,
            permutation_views=bool(cfg.train.operator_bank.permutation_views),
            canonical_probability=float(cfg.train.operator_bank.canonical_probability),
            hot_shards=int(cfg.train.operator_bank.hot_shards),
            expected_pair_manifest_sha256=str(cfg.train.operator_bank.pair_manifest_sha256),
            rank=0,
            world_size=1,
            max_active_strata=int(cfg.train.operator_bank.max_active_strata),
            max_active_bundle_bytes=int(cfg.train.operator_bank.max_active_bundle_bytes),
            logger=logger,
        ) as (dataset, sampler):
            sampler.set_start_index(start_index)
            bundle_iter = (dataset[request] for request in sampler)
            mixer = BalancedOperatorBankMixer(dataset, bundle_iter, start_index=start_index)
            for batch_idx in range(int(args.batches)):
                cached_batches.append(
                    _fetch_presliced_training_batch_cpu(
                        dataset_iter=mixer,
                        batch_size=int(args.batch_size),
                        logger=logger,
                    )
                )

        original_encoder_weight_post_norms = model.encoder_weight_post_norms
        original_encoder_latent_post_norms = model.encoder_latent_post_norms
        original_decoder_post_norms = model.decoder_post_norms
        for postnorm_mode in postnorm_modes:
            _set_postnorm_mode(
                model,
                postnorm_mode,
                encoder_weight_post_norms=original_encoder_weight_post_norms,
                encoder_latent_post_norms=original_encoder_latent_post_norms,
                decoder_post_norms=original_decoder_post_norms,
            )
            for batch_idx, batch in enumerate(cached_batches):
                tensors = [batch.W, batch.x, batch.x_mask, batch.d_in_mask, batch.d_out_mask]
                W, x, x_mask, d_in_mask, d_out_mask = [tensor.to(device) for tensor in tensors]
                trace.clear()
                model.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                grad_context = torch.no_grad() if args.interventions_only else contextlib.nullcontext()
                with autocast, grad_context:
                    total, loss_details, outputs = _losses(model, cfg, W, x, x_mask, d_in_mask, d_out_mask)
                W_hat, mu, logvar, pred_dirs, debug_info = outputs
                posterior = {
                    "mu_rms": float(mu.detach().float().square().mean().sqrt().item()),
                    "mu_batch_variance_mean": float(mu.detach().float().var(dim=0, unbiased=False).mean().item()),
                    "mu_active_fraction_var_gt_1e-4": float(
                        (mu.detach().float().var(dim=0, unbiased=False) > 1e-4).float().mean().item()
                    ),
                    "posterior_std_mean": float(torch.exp(0.5 * logvar.detach().float()).mean().item()),
                    "decoder_z_rms": float(
                        debug_info["latent_decoder_z"].detach().float().square().mean().sqrt().item()
                    ),
                    "base_z_rms": float(debug_info["latent_base_z"].detach().float().square().mean().sqrt().item()),
                }
                boundary_component_gradients = (
                    None
                    if args.interventions_only
                    else _boundary_component_gradients(
                        reconstruction=loss_details["reconstruction"],
                        weighted_kl=loss_details["weighted_kl"],
                        mu=mu,
                        debug_info=debug_info,
                    )
                )
                if args.skip_backward or args.interventions_only:
                    parameter_groups = []
                else:
                    total.backward()
                    parameter_rows = _parameter_gradient_rows(model)
                    parameter_groups = _aggregate_parameter_groups(parameter_rows)
                for row in trace.rows:
                    row.update(
                        {"batch": batch_idx, "checkpoint_step": checkpoint_step, "postnorm_mode": postnorm_mode}
                    )
                for row in parameter_groups:
                    row.update(
                        {"batch": batch_idx, "checkpoint_step": checkpoint_step, "postnorm_mode": postnorm_mode}
                    )
                all_trace_rows.extend(trace.rows)
                all_parameter_groups.extend(parameter_groups)
                batch_summary = {
                    "batch": batch_idx,
                    "postnorm_mode": postnorm_mode,
                    "logical_indices": list(batch.logical_indices),
                    "losses": {key: float(value.detach().item()) for key, value in loss_details.items()},
                    "posterior": posterior,
                    "boundary_component_gradients": boundary_component_gradients,
                    "gpu_memory_allocated_mib": (
                        float(torch.cuda.memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
                    ),
                }
                if int(W.shape[0]) >= 2:
                    with torch.no_grad():
                        batch_summary["latent_interventions"] = _latent_intervention_losses(
                            model,
                            cfg,
                            W=W,
                            x=x,
                            x_mask=x_mask,
                            d_in_mask=d_in_mask,
                            d_out_mask=d_out_mask,
                            debug_info=debug_info,
                            mu=mu,
                            logvar=logvar,
                            canonical_W_hat=W_hat,
                            canonical_pred_dirs=pred_dirs,
                        )
                batch_summaries.append(batch_summary)
                logger.info(
                    "postnorm_mode=%s batch=%s total_loss=%.6f",
                    postnorm_mode,
                    batch_idx,
                    batch_summary["losses"]["total"],
                )
                del total, loss_details, outputs, W_hat, mu, logvar, pred_dirs, debug_info
                del W, x, x_mask, d_in_mask, d_out_mask
                model.zero_grad(set_to_none=True)
    finally:
        trace.close()

    _write_csv(output_dir / "activation_gradient_trace.csv", all_trace_rows)
    _write_csv(output_dir / "parameter_gradient_groups.csv", all_parameter_groups)
    report = {
        "schema": "weightclip_ae_gradient_path_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "amp_dtype": "torch.bfloat16" if device.type == "cuda" else "float32",
        "sampling_gate": sampling_gate,
        "kl_beta": float(cfg.train.kl_beta),
        "skip_backward": bool(args.skip_backward),
        "interventions_only": bool(args.interventions_only),
        "batches": batch_summaries,
        "start_index": start_index,
        "postnorm_modes": postnorm_modes,
        "artifacts": {
            "activation_gradient_trace_csv": str(output_dir / "activation_gradient_trace.csv"),
            "parameter_gradient_groups_csv": str(output_dir / "parameter_gradient_groups.csv"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Gradient-path artifacts written: %s", output_dir)


if __name__ == "__main__":
    main()
