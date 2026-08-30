from __future__ import annotations

import argparse
import contextlib
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer
from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from big_vae.models.vae_shared import _apply_sequence_mask, _decode_direction_and_logscale
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.big_vae.two_operator_overfit import two_operator_data_pipeline
from training.weightclip_benchmark.analyze_ae_two_operator_progressive_bypass import (
    _atomic_json,
    _resolve_checkpoint_loss,
    _swap_indices,
    _train_arm,
)


_SCHEMA = "weightclip_ae_two_operator_decoder_ladder_v1"
_HARD_MAX_STEPS = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded two-operator ladder localizing V5 decoder stack, q-norm, and frozen heads."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v5_two_operator_decoder_ladder.yaml"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_decoder_ladder_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"config must use schema {_SCHEMA!r}")
    allowed = {
        "schema",
        "output_root",
        "seed",
        "max_steps",
        "eval_every_steps",
        "log_every_steps",
        "learning_rate",
        "eps_sensitivity",
        "prefix_adam_eps",
        "prefix_depths",
        "loss",
        "early_success",
        "selected_operators",
    }
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"decoder ladder config mismatch: unknown={unknown} missing={missing}")
    steps = int(payload["max_steps"])
    if not 1 <= steps <= _HARD_MAX_STEPS:
        raise ValueError(f"max_steps must be in [1,{_HARD_MAX_STEPS}], got {steps}")
    if float(payload["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    sensitivity = [float(value) for value in payload["eps_sensitivity"]]
    if sensitivity != [1.0e-8, 1.0e-12]:
        raise ValueError("eps_sensitivity must be exactly [1e-8,1e-12]")
    if float(payload["prefix_adam_eps"]) != 1.0e-12:
        raise ValueError("prefix ladder must use floor-free Adam eps=1e-12")
    if [int(value) for value in payload["prefix_depths"]] != list(range(9)):
        raise ValueError("prefix_depths must enumerate every V5 decoder prefix 0..8")
    loss = payload["loss"]
    if loss != {
        "source": "checkpoint.train.struct_loss",
        "expected_lambda_dir": 1.0,
        "expected_lambda_scale": 0.1,
    }:
        raise ValueError("loss must bind exact checkpoint structural dir=1/scale=.1 objective")
    selected = payload["selected_operators"]
    if not isinstance(selected, list) or len(selected) != 2:
        raise ValueError("selected_operators must contain exactly two entries")
    identities = []
    for row in selected:
        if not isinstance(row, Mapping):
            raise TypeError("selected operator entries must be mappings")
        sha = str(row.get("checkpoint_sha256", ""))
        layer = str(row.get("layer_key", ""))
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha) or not layer:
            raise ValueError(f"invalid selected operator identity: {row}")
        identities.append((sha, layer))
    if len(set(identities)) != 2 or identities[0][1] != identities[1][1]:
        raise ValueError("selected operators must be distinct checkpoints of the same layer")
    if steps * len(_arm_plan(payload)) > 2_000:
        raise ValueError("aggregate decoder-ladder optimizer-step budget must not exceed 2,000")
    return payload


class ScaledDeltaCodeBank(nn.Module):
    """Independent dimensionless deltas around exact per-tile checkpoint states."""

    def __init__(self, initial: torch.Tensor) -> None:
        super().__init__()
        if initial.ndim != 3 or not torch.isfinite(initial).all():
            raise ValueError(f"initial code must be finite [tile,token,dim], got {tuple(initial.shape)}")
        self.output_dtype = initial.dtype
        base = initial.detach().float().clone()
        scale = base.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1.0e-6)
        self.register_buffer("base", base)
        self.register_buffer("scale", scale)
        self.delta = nn.Parameter(torch.zeros_like(base))

    def select(self, indices: torch.Tensor) -> torch.Tensor:
        value = self.base.index_select(0, indices) + self.scale.index_select(
            0, indices
        ) * self.delta.index_select(0, indices)
        return value.to(dtype=self.output_dtype)


class ScaledHeadLogitBank(nn.Module):
    """Independent raw direction/scale logits with separate physical scales."""

    def __init__(self, initial_u: torch.Tensor, initial_s: torch.Tensor) -> None:
        super().__init__()
        if initial_u.ndim != 3 or initial_s.ndim != 3 or initial_s.shape[-1] != 1:
            raise ValueError("raw head logits must be u=[B,Q,p], s=[B,Q,1]")
        if initial_u.shape[:2] != initial_s.shape[:2]:
            raise ValueError("raw direction and scale logits must share B,Q")
        self.output_u_dtype = initial_u.dtype
        self.output_s_dtype = initial_s.dtype
        self.register_buffer("base_u", initial_u.detach().float().clone())
        self.register_buffer("base_s", initial_s.detach().float().clone())
        self.register_buffer(
            "scale_u",
            self.base_u.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1.0e-6),
        )
        self.register_buffer(
            "scale_s",
            self.base_s.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1.0e-6),
        )
        self.delta = nn.Parameter(
            torch.zeros((*initial_u.shape[:2], int(initial_u.shape[2]) + 1), dtype=torch.float32)
        )

    def select(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.delta.index_select(0, indices)
        u = self.base_u.index_select(0, indices) + self.scale_u.index_select(0, indices) * delta[..., :-1]
        s = self.base_s.index_select(0, indices) + self.scale_s.index_select(0, indices) * delta[..., -1:]
        return u.to(dtype=self.output_u_dtype), s.to(dtype=self.output_s_dtype)


def decode_from_normalized_head_input(
    model: nn.Module,
    q_dir: torch.Tensor,
    *,
    z: torch.Tensor,
    q_pos_emb: torch.Tensor,
    d_in: int,
    d_out: int,
    d_in_pad: int,
    T: int,
    patch_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    batch, queries, _width = q_dir.shape
    patch_size = int(model.cfg.patch_size)
    query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(
        batch, queries
    )
    q_dir = _apply_sequence_mask(q_dir, query_mask)
    u_hat = _apply_sequence_mask(model.direction_head(q_dir), query_mask).reshape(batch * queries, patch_size)
    s_hat = _apply_sequence_mask(model.scale_head(q_dir), query_mask).reshape(batch * queries)
    flat = _decode_direction_and_logscale(
        u_hat=u_hat,
        s_hat=s_hat,
        eps=float(model.output_eps),
        s_min=float(model.output_s_min),
        s_max=float(model.output_s_max),
    )
    patches = flat.view(batch, d_out, T, patch_size)
    prediction = patches.view(batch, d_out, d_in_pad).transpose(1, 2)[:, :d_in, :]
    prediction = prediction * d_in_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    prediction = prediction * d_out_mask.to(dtype=prediction.dtype).unsqueeze(1)
    # z/q_pos_emb are deliberately unused: accepting them documents and tests
    # that this arm bypasses both latent shortcut and q-norm, not the heads.
    del z, q_pos_emb
    return prediction


def decode_from_raw_head_logits(
    model: nn.Module,
    u_hat: torch.Tensor,
    s_hat: torch.Tensor,
    *,
    d_in: int,
    d_out: int,
    d_in_pad: int,
    T: int,
    patch_mask: torch.Tensor,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    batch, queries, patch_size = u_hat.shape
    if int(patch_size) != int(model.cfg.patch_size) or tuple(s_hat.shape) != (batch, queries, 1):
        raise ValueError("raw head logit shapes disagree with model patch geometry")
    query_mask = (patch_mask.unsqueeze(1).expand(-1, d_out, -1) & d_out_mask.unsqueeze(-1)).reshape(
        batch, queries
    )
    u_flat = _apply_sequence_mask(u_hat, query_mask).reshape(batch * queries, patch_size)
    s_flat = _apply_sequence_mask(s_hat, query_mask).reshape(batch * queries)
    decoded = _decode_direction_and_logscale(
        u_hat=u_flat,
        s_hat=s_flat,
        eps=float(model.output_eps),
        s_min=float(model.output_s_min),
        s_max=float(model.output_s_max),
    )
    patches = decoded.view(batch, d_out, T, patch_size)
    prediction = patches.view(batch, d_out, d_in_pad).transpose(1, 2)[:, :d_in, :]
    prediction = prediction * d_in_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    prediction = prediction * d_out_mask.to(dtype=prediction.dtype).unsqueeze(1)
    return prediction


def _arm_plan(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for eps in spec["eps_sensitivity"]:
        suffix = "eps1e8" if float(eps) == 1.0e-8 else "eps1e12"
        rows.extend(
            [
                {"kind": "normalized_head_input", "name": f"normalized_head_input_{suffix}", "eps": float(eps)},
                {"kind": "raw_pre_qnorm", "name": f"raw_pre_qnorm_{suffix}", "eps": float(eps)},
                {"kind": "raw_head_logits", "name": f"raw_head_logits_{suffix}", "eps": float(eps)},
            ]
        )
        if float(eps) == 1.0e-8:
            rows.append({"kind": "decoder_cut", "name": "decoder_cut_0_eps1e8", "eps": float(eps), "depth": 0})
    rows.extend(
        {
            "kind": "decoder_cut",
            "name": f"decoder_cut_{depth}_eps1e12",
            "eps": float(spec["prefix_adam_eps"]),
            "depth": int(depth),
        }
        for depth in spec["prefix_depths"]
    )
    return rows


def main() -> None:
    args = _parse_args()
    spec = load_decoder_ladder_spec(args.config.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = Path(str(spec["output_root"])).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"decoder-ladder output root already exists: {output_root}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("ae-two-op-decoder-ladder")
    logger.info(
        "stage=startup checkpoint=%s device=%s dtype=bf16 seed=%s max_steps=%s output=%s",
        checkpoint_path,
        args.device,
        spec["seed"],
        spec["max_steps"],
        output_root,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    if int(checkpoint["step"]) != 2_500:
        raise ValueError(f"decoder ladder requires exact two-op step2500 checkpoint, got {checkpoint['step']}")
    if str(cfg.model.big_vae.architecture_version) != "latent_mandatory_bridge_prenorm_v5":
        raise ValueError("decoder ladder requires the exact V5 architecture")
    if bool(cfg.model.big_vae.use_latent_sampling) or bool(cfg.model.big_vae.use_encoder_mu_head):
        raise ValueError("decoder ladder requires deterministic latent configuration")
    loss_cfg = _resolve_checkpoint_loss(cfg, spec)
    pair_cfg = cfg.train.operator_bank
    plan = _arm_plan(spec)
    contract = {
        "schema": _SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": 2_500,
        "device": str(args.device),
        "dtype": "bf16-autocast",
        "seed": int(spec["seed"]),
        "max_steps_per_arm": int(spec["max_steps"]),
        "hard_max_steps_per_arm": _HARD_MAX_STEPS,
        "hard_max_aggregate_optimizer_steps": 2_000,
        "planned_max_aggregate_optimizer_steps": int(spec["max_steps"]) * len(plan),
        "loss": loss_cfg,
        "parameterization": "code=frozen_per_tile_start+per_tile_rms*dimensionless_delta",
        "independent_code_count": 18,
        "swap": "exact_A_i_to_B_i",
        "arm_plan": plan,
        "pair_manifest": str(pair_cfg.pair_manifest),
        "pair_manifest_sha256": str(pair_cfg.pair_manifest_sha256),
        "selected_operators": spec["selected_operators"],
        "output_root": str(output_root),
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    output_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_root / "resolved_contract.json", contract)
    torch.manual_seed(int(spec["seed"]))
    device = torch.device(args.device)
    logger.info("stage=load_data pair=%s cache=immutable-operator-bank-canonical", pair_cfg.pair_manifest)
    with two_operator_data_pipeline(
        pair_cfg.pair_manifest,
        selected_operators=spec["selected_operators"],
        seed=int(spec["seed"]),
        hot_shards=int(pair_cfg.hot_shards),
        expected_pair_manifest_sha256=str(pair_cfg.pair_manifest_sha256),
        max_active_strata=int(pair_cfg.max_active_strata),
        max_active_bundle_bytes=int(pair_cfg.max_active_bundle_bytes),
    ) as (dataset, sampler):
        sampler.set_start_index(0)
        mixer = BalancedOperatorBankMixer(dataset, (dataset[request] for request in sampler), start_index=0)
        batch = _fetch_presliced_training_batch_cpu(dataset_iter=mixer, batch_size=18, logger=logger)
        assignments, _views = dataset.source._cycle_plan(0)
        if tuple(batch.logical_indices) != tuple(range(18)) or len(set(assignments)) != 18:
            raise RuntimeError("ladder data is not the exact 18-tile A_i/B_i cycle")
        swap = _swap_indices(assignments)
        target = batch.W.to(device)
        x = batch.x.to(device)
        x_mask = batch.x_mask.to(device)
        d_in_mask = batch.d_in_mask.to(device)
        d_out_mask = batch.d_out_mask.to(device)
        logger.info("stage=build_model checkpoint_step=2500")
        model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        del checkpoint
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        frozen_versions = tuple((name, parameter._version) for name, parameter in model.named_parameters())

        def amp_context() -> contextlib.AbstractContextManager[Any]:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

        with torch.no_grad(), amp_context():
            initial_prediction, _mu, _logvar, initial_dirs, debug = model.forward_debug(
                target, x, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
            )
            z_actual = debug["latent_decoder_z"].view(
                18, int(model.cfg.big_vae.num_latents), int(model.cfg.big_vae.d_lat)
            ).float()
            q_base, q_pos_emb, q_pos_o, q_pos_t = model._build_decoder_query_state(
                batch_size=18,
                dist_patch_by_patch=debug["dist_patch_by_patch"],
                d_out=int(target.shape[2]),
                T=int(debug["T"]),
            )
            bridge_start = model.mandatory_latent_bridge(
                q_base, z_actual.to(q_base.dtype), query_mask=debug["decoder_query_mask"]
            )
            prefix_states = [bridge_start]
            state = bridge_start
            for layer in model.decoder_layers:
                state = layer(state, q_pos=q_pos_o, q_pos2=q_pos_t, q_mask=debug["decoder_query_mask"])
                prefix_states.append(state)
            raw_pre_qnorm = prefix_states[-1]
            normalized_head_input = _apply_sequence_mask(
                model.q_tokens_norm(raw_pre_qnorm), debug["decoder_query_mask"]
            )
            raw_u_hat = _apply_sequence_mask(
                model.direction_head(normalized_head_input), debug["decoder_query_mask"]
            )
            raw_s_hat = _apply_sequence_mask(
                model.scale_head(normalized_head_input), debug["decoder_query_mask"]
            )
        if len(prefix_states) != 9 or not all(
            bool(value.all()) for value in (d_in_mask, d_out_mask, debug["patch_mask"])
        ):
            raise RuntimeError("decoder ladder requires 8 layers and fully valid 128x128 tiles")
        reconstructed_loss = WeightQuantileVAE.patch_structure_loss(
            target,
            initial_prediction,
            patch_size=int(loss_cfg["patch_size"]),
            gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]),
            lambda_scale=float(loss_cfg["lambda_scale"]),
            lambda_rec=0.0,
            lambda_rel=0.0,
            huber_delta=float(loss_cfg["huber_delta"]),
        )[0]
        production_loss = WeightQuantileVAE.patch_structure_loss(
            target,
            initial_prediction,
            patch_size=int(loss_cfg["patch_size"]),
            gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]),
            lambda_scale=float(loss_cfg["lambda_scale"]),
            lambda_rec=0.0,
            lambda_rel=0.0,
            huber_delta=float(loss_cfg["huber_delta"]),
            pred_dirs=initial_dirs,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
        )[0]
        parity_error = abs(float(reconstructed_loss.float()) - float(production_loss.float()))
        if parity_error > 1.0e-5:
            raise RuntimeError(f"structural loss path parity failed: {parity_error}")
        contract["loss_path_parity"] = {"absolute_error": parity_error, "tolerance": 1.0e-5}
        contract["initial_code_rms"] = {
            "normalized_head_input": float(normalized_head_input.float().square().mean().sqrt()),
            "raw_pre_qnorm": float(raw_pre_qnorm.float().square().mean().sqrt()),
            "bridge_start": float(bridge_start.float().square().mean().sqrt()),
        }
        contract["captured_native_dtypes"] = {
            "normalized_head_input": str(normalized_head_input.dtype),
            "raw_pre_qnorm": str(raw_pre_qnorm.dtype),
            "bridge_start": str(bridge_start.dtype),
            "raw_u_hat": str(raw_u_hat.dtype),
            "raw_s_hat": str(raw_s_hat.dtype),
        }
        _atomic_json(output_root / "resolved_contract.json", contract)

        def decode_qdir(code: torch.Tensor) -> torch.Tensor:
            with amp_context():
                return decode_from_normalized_head_input(
                    model,
                    code,
                    z=z_actual.flatten(1),
                    q_pos_emb=q_pos_emb,
                    d_in=int(target.shape[1]),
                    d_out=int(target.shape[2]),
                    d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]),
                    patch_mask=debug["patch_mask"],
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                )

        def decode_raw(code: torch.Tensor) -> torch.Tensor:
            with amp_context():
                return model._decode_query_tokens_to_output(
                    code,
                    z=z_actual.flatten(1),
                    q_pos_emb=q_pos_emb,
                    d_in=int(target.shape[1]),
                    d_out=int(target.shape[2]),
                    d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]),
                    patch_mask=debug["patch_mask"],
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    disable_z_shortcut=True,
                )[0]

        def decode_suffix(code: torch.Tensor, cut_depth: int) -> torch.Tensor:
            with amp_context():
                state = code
                for layer in model.decoder_layers[cut_depth:]:
                    state = layer(state, q_pos=q_pos_o, q_pos2=q_pos_t, q_mask=debug["decoder_query_mask"])
                return model._decode_query_tokens_to_output(
                    state,
                    z=z_actual.flatten(1),
                    q_pos_emb=q_pos_emb,
                    d_in=int(target.shape[1]),
                    d_out=int(target.shape[2]),
                    d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]),
                    patch_mask=debug["patch_mask"],
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    disable_z_shortcut=True,
                )[0]

        def decode_logits(logits: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            u_hat, s_hat = logits
            with amp_context():
                return decode_from_raw_head_logits(
                    model,
                    u_hat,
                    s_hat,
                    d_in=int(target.shape[1]),
                    d_out=int(target.shape[2]),
                    d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]),
                    patch_mask=debug["patch_mask"],
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                )

        with torch.no_grad():
            parity_predictions = {
                "post_qnorm": decode_qdir(normalized_head_input),
                "post_stack_pre_qnorm": decode_raw(raw_pre_qnorm),
                "post_head_logits": decode_logits((raw_u_hat, raw_s_hat)),
            }
            parity_predictions.update(
                {
                    f"decoder_cut_{depth}": decode_suffix(prefix_states[depth], depth)
                    for depth in range(9)
                }
            )
            cutpoint_parity = {
                name: {
                    "prediction_max_abs": float((prediction.float() - initial_prediction.float()).abs().max()),
                    "prediction_rms": float((prediction.float() - initial_prediction.float()).square().mean().sqrt()),
                }
                for name, prediction in parity_predictions.items()
            }
        if max(row["prediction_max_abs"] for row in cutpoint_parity.values()) > 1.0e-5:
            raise RuntimeError(f"delta-zero cutpoints do not reproduce checkpoint output: {cutpoint_parity}")
        contract["delta_zero_cutpoint_parity"] = cutpoint_parity
        _atomic_json(output_root / "resolved_contract.json", contract)

        results = []
        for row in plan:
            if row["kind"] == "normalized_head_input":
                start = normalized_head_input
                decoder: Callable[[torch.Tensor], torch.Tensor] = decode_qdir
            elif row["kind"] == "raw_pre_qnorm":
                start = raw_pre_qnorm
                decoder = decode_raw
            elif row["kind"] == "raw_head_logits":
                start = None
                decoder = decode_logits
            else:
                depth = int(row["depth"])
                start = prefix_states[depth]
                decoder = partial(decode_suffix, cut_depth=depth)
            arm = (
                ScaledHeadLogitBank(raw_u_hat, raw_s_hat).to(device)
                if row["kind"] == "raw_head_logits"
                else ScaledDeltaCodeBank(start).to(device)
            )
            logger.info(
                "stage=arm_start arm=%s eps=%.1e code_rms=%.6f trainable=%d",
                row["name"],
                row["eps"],
                float(arm.scale_u.mean()) if isinstance(arm, ScaledHeadLogitBank) else float(arm.scale.mean()),
                arm.delta.numel(),
            )
            result = _train_arm(
                name=str(row["name"]),
                module=arm,
                predict=lambda indices, module=arm, fn=decoder: fn(module.select(indices)),
                target=target,
                swap=swap,
                assignments=assignments,
                source=dataset.source,
                spec=spec,
                loss_cfg=loss_cfg,
                output_dir=output_root / str(row["name"]),
                logger=logger,
                learning_rate=float(spec["learning_rate"]),
                code_parameter=arm.delta,
                adam_eps=float(row["eps"]),
            )
            result["ladder_arm"] = row
            result["initial_code_rms_mean"] = (
                {
                    "u": float(arm.scale_u.mean()),
                    "s": float(arm.scale_s.mean()),
                }
                if isinstance(arm, ScaledHeadLogitBank)
                else float(arm.scale.mean())
            )
            results.append(result)
        final_versions = tuple((name, parameter._version) for name, parameter in model.named_parameters())
        if final_versions != frozen_versions:
            raise RuntimeError("frozen production model parameter versions changed during decoder ladder")
        summary = {
            "schema": "weightclip_ae_two_operator_decoder_ladder_summary_v1",
            "contract": contract,
            "frozen_model_parameter_versions_unchanged": True,
            "arms": results,
        }
        _atomic_json(output_root / "summary.json", summary)
        logger.info("stage=complete summary=%s", output_root / "summary.json")


if __name__ == "__main__":
    main()
