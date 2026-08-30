from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer
from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.big_vae.two_operator_overfit import (
    _loss_metrics,
    _nrmse,
    _stitch,
    two_operator_data_pipeline,
)


_SCHEMA = "weightclip_ae_two_operator_progressive_bypass_v1"
_HARD_MAX_STEPS = 2_000
_ARMS = ("direct_hidden", "learnable_z", "learnable_z_plus_qpos", "encoder_linear_readout")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four bounded two-operator bypass arms against one frozen V5 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v5_two_operator_progressive_bypass.yaml"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_progressive_bypass_spec(path: Path) -> dict[str, Any]:
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
        "learning_rates",
        "loss",
        "early_success",
        "selected_operators",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown progressive-bypass config keys: {unknown}")
    missing = sorted(allowed - set(payload))
    if missing:
        raise ValueError(f"missing progressive-bypass config keys: {missing}")
    steps = int(payload["max_steps"])
    if not 1 <= steps <= _HARD_MAX_STEPS:
        raise ValueError(f"max_steps must be in [1,{_HARD_MAX_STEPS}], got {steps}")
    if int(payload["eval_every_steps"]) < 1 or int(payload["log_every_steps"]) < 1:
        raise ValueError("eval/log intervals must be positive")
    rates = payload["learning_rates"]
    if not isinstance(rates, dict) or set(rates) != set(_ARMS) or any(float(rates[name]) <= 0 for name in _ARMS):
        raise ValueError(f"learning_rates must contain positive values for exactly {_ARMS}")
    loss = payload["loss"]
    loss_keys = {"source", "expected_lambda_dir", "expected_lambda_scale"}
    if not isinstance(loss, dict) or set(loss) != loss_keys:
        raise ValueError(f"loss must contain exactly {sorted(loss_keys)}")
    if str(loss["source"]) != "checkpoint.train.struct_loss":
        raise ValueError("structural loss must be read from checkpoint.train.struct_loss")
    if float(loss["expected_lambda_dir"]) != 1.0 or float(loss["expected_lambda_scale"]) != 0.1:
        raise ValueError("progressive bypass requires the same structural dir=1.0 scale=0.1 objective")
    early = payload["early_success"]
    early_keys = {
        "enabled",
        "min_step",
        "consecutive_evals",
        "tile_dir_max",
        "tile_scale_max",
        "operator_nrmse_max",
        "swap_total_delta_min",
    }
    if not isinstance(early, dict) or set(early) != early_keys:
        raise ValueError(f"early_success must contain exactly {sorted(early_keys)}")
    selected = payload["selected_operators"]
    if not isinstance(selected, list) or len(selected) != 2:
        raise ValueError("selected_operators must contain exactly two entries")
    identities = []
    for row in selected:
        if not isinstance(row, Mapping):
            raise TypeError("selected operator rows must be mappings")
        sha = str(row.get("checkpoint_sha256", ""))
        layer = str(row.get("layer_key", ""))
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha) or not layer:
            raise ValueError(f"invalid selected operator identity: {row}")
        identities.append((sha, layer))
    if len(set(identities)) != 2 or identities[0][1] != identities[1][1]:
        raise ValueError("selected operators must be distinct checkpoints of the same layer")
    return payload


def _swap_indices(assignments: Sequence[tuple[tuple[str, str], int]]) -> torch.Tensor:
    lookup = {assignment: index for index, assignment in enumerate(assignments)}
    keys = tuple(dict.fromkeys(key for key, _local in assignments))
    if len(keys) != 2:
        raise ValueError("swap requires exactly two operator keys")
    swapped = [lookup[(keys[1] if key == keys[0] else keys[0], local)] for key, local in assignments]
    if sorted(swapped) != list(range(len(assignments))) or any(swapped[swapped[index]] != index for index in range(len(swapped))):
        raise RuntimeError("A_i<->B_i swap must be a complete involution")
    return torch.tensor(swapped, dtype=torch.long)


class MeanLatentLinearReadout(nn.Module):
    """One shared low-capacity bias-free map; no X, identity, or per-operator head."""

    def __init__(self, d_lat: int, output_shape: tuple[int, int]) -> None:
        super().__init__()
        self.readout = nn.Linear(int(d_lat), int(math.prod(output_shape)), bias=False)
        nn.init.zeros_(self.readout.weight)
        self.output_shape = tuple(int(value) for value in output_shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or int(z.shape[-1]) != int(self.readout.in_features):
            raise ValueError(f"z must be [B,L,{self.readout.in_features}], got {tuple(z.shape)}")
        features = z.float().mean(dim=1)
        return self.readout(features).view(z.shape[0], *self.output_shape)


class _DirectHiddenArm(nn.Module):
    def __init__(self, initial_hidden: torch.Tensor) -> None:
        super().__init__()
        self.hidden = nn.Parameter(initial_hidden.detach().float().clone())

    def select(self, indices: torch.Tensor) -> torch.Tensor:
        return self.hidden.index_select(0, indices)


class _LearnableZArm(nn.Module):
    def __init__(self, initial_z: torch.Tensor) -> None:
        super().__init__()
        self.z = nn.Parameter(initial_z.detach().float().clone())

    def select(self, indices: torch.Tensor) -> torch.Tensor:
        return self.z.index_select(0, indices)


def _structural_loss(target: torch.Tensor, prediction: torch.Tensor, loss_cfg: Mapping[str, Any]) -> torch.Tensor:
    total, _details = WeightQuantileVAE.patch_structure_loss(
        target,
        prediction,
        patch_size=int(loss_cfg["patch_size"]),
        gamma=float(loss_cfg["gamma"]),
        lambda_dir=float(loss_cfg["lambda_dir"]),
        lambda_scale=float(loss_cfg["lambda_scale"]),
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=float(loss_cfg["huber_delta"]),
    )
    return total


def _resolve_checkpoint_loss(cfg: Any, spec: Mapping[str, Any]) -> dict[str, float | int]:
    row = cfg.train.struct_loss
    resolved: dict[str, float | int] = {
        "patch_size": int(cfg.model.patch_size),
        "gamma": float(row.gamma),
        "lambda_dir": float(row.lambda_dir),
        "lambda_scale": float(row.lambda_scale),
        "huber_delta": float(row.huber_delta),
    }
    actual_contract = {
        "lambda_dir": resolved["lambda_dir"],
        "lambda_scale": resolved["lambda_scale"],
        "lambda_rec": float(row.lambda_rec),
        "lambda_rel": float(row.lambda_rel),
    }
    expected_contract = {
        "lambda_dir": float(spec["loss"]["expected_lambda_dir"]),
        "lambda_scale": float(spec["loss"]["expected_lambda_scale"]),
        "lambda_rec": 0.0,
        "lambda_rel": 0.0,
    }
    if actual_contract != expected_contract:
        raise ValueError(f"checkpoint structural objective drifted: expected={expected_contract} actual={actual_contract}")
    return resolved


def _code_gradient_stats(
    parameter: torch.Tensor, *, adam_eps: float, learning_rate: float
) -> dict[str, Any]:
    if parameter.grad is None or parameter.grad.ndim < 2:
        raise RuntimeError("lookup code parameter must have a per-tile gradient")
    rows = parameter.grad.detach().float().flatten(1)
    norms = rows.norm(dim=1)
    summed = rows.sum(dim=0).norm()
    denominator = norms.sum().clamp_min(1.0e-30)
    pair_cosines = torch.nn.functional.cosine_similarity(rows[0::2], rows[1::2], dim=1)
    first_step_fraction = rows.abs() / (rows.abs() + float(adam_eps))
    return {
        "code_grad_rms_by_tile": [float(row.square().mean().sqrt().item()) for row in rows],
        "code_grad_norm_by_tile": [float(value.item()) for value in norms],
        "hypothetical_shared_code_cancellation_ratio": float((summed / denominator).item()),
        "paired_Ai_Bi_code_grad_cosines": [float(value.item()) for value in pair_cosines],
        "adam_eps": float(adam_eps),
        "adam_first_step_normalization_fraction_mean": float(first_step_fraction.mean().item()),
        "adam_first_step_normalization_fraction_p50": float(first_step_fraction.median().item()),
        "adam_first_step_effective_update_rms": float(
            float(learning_rate) * first_step_fraction.square().mean().sqrt().item()
        ),
    }


def _metrics(
    *,
    target: torch.Tensor,
    matched: torch.Tensor,
    swapped: torch.Tensor,
    assignments: Sequence[tuple[tuple[str, str], int]],
    source: Any,
    loss_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {"tiles": []}
    for index, (key, local_index) in enumerate(assignments):
        matched_row = _loss_metrics(
            target[index : index + 1], matched[index : index + 1],
            patch_size=int(loss_cfg["patch_size"]), gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]), lambda_scale=float(loss_cfg["lambda_scale"]),
            huber_delta=float(loss_cfg["huber_delta"]),
        )
        swapped_row = _loss_metrics(
            target[index : index + 1], swapped[index : index + 1],
            patch_size=int(loss_cfg["patch_size"]), gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]), lambda_scale=float(loss_cfg["lambda_scale"]),
            huber_delta=float(loss_cfg["huber_delta"]),
        )
        output["tiles"].append({
            "tile_index": index,
            "operator_index": 0 if key == source._keys[0] else 1,
            "local_tile_index": int(local_index),
            **{f"matched_{name}": value for name, value in matched_row.items()},
            **{f"swapped_{name}": value for name, value in swapped_row.items()},
            "matched_nrmse": _nrmse(target[index], matched[index]),
            "swapped_nrmse": _nrmse(target[index], swapped[index]),
            "swap_total_delta": swapped_row["total"] - matched_row["total"],
        })
    output["operators"] = []
    for operator_index, key in enumerate(source._keys):
        target_matrix = _stitch(target, assignments, source, key)
        matched_matrix = _stitch(matched, assignments, source, key)
        swapped_matrix = _stitch(swapped, assignments, source, key)
        matched_row = _loss_metrics(
            target_matrix, matched_matrix,
            patch_size=int(loss_cfg["patch_size"]), gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]), lambda_scale=float(loss_cfg["lambda_scale"]),
            huber_delta=float(loss_cfg["huber_delta"]),
        )
        swapped_row = _loss_metrics(
            target_matrix, swapped_matrix,
            patch_size=int(loss_cfg["patch_size"]), gamma=float(loss_cfg["gamma"]),
            lambda_dir=float(loss_cfg["lambda_dir"]), lambda_scale=float(loss_cfg["lambda_scale"]),
            huber_delta=float(loss_cfg["huber_delta"]),
        )
        output["operators"].append({
            "operator_index": operator_index,
            "checkpoint_sha256": key[0],
            "layer_key": key[1],
            **{f"matched_{name}": value for name, value in matched_row.items()},
            **{f"swapped_{name}": value for name, value in swapped_row.items()},
            "matched_nrmse": _nrmse(target_matrix, matched_matrix),
            "swapped_nrmse": _nrmse(target_matrix, swapped_matrix),
            "swap_total_delta": swapped_row["total"] - matched_row["total"],
        })
    output["matched_max_tile_dir"] = max(row["matched_dir"] for row in output["tiles"])
    output["matched_max_tile_scale"] = max(row["matched_scale"] for row in output["tiles"])
    output["matched_max_operator_nrmse"] = max(row["matched_nrmse"] for row in output["operators"])
    output["swap_min_operator_total_delta"] = min(row["swap_total_delta"] for row in output["operators"])
    return output


def _success(metrics: Mapping[str, Any], early: Mapping[str, Any]) -> bool:
    return (
        float(metrics["matched_max_tile_dir"]) <= float(early["tile_dir_max"])
        and float(metrics["matched_max_tile_scale"]) <= float(early["tile_scale_max"])
        and float(metrics["matched_max_operator_nrmse"]) <= float(early["operator_nrmse_max"])
        and float(metrics["swap_min_operator_total_delta"]) >= float(early["swap_total_delta_min"])
    )


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _train_arm(
    *,
    name: str,
    module: nn.Module,
    predict: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    swap: torch.Tensor,
    assignments: Sequence[tuple[tuple[str, str], int]],
    source: Any,
    spec: Mapping[str, Any],
    loss_cfg: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    learning_rate: float,
    train_indices: torch.Tensor | None = None,
    code_parameter: torch.Tensor | None = None,
    adam_eps: float = 1.0e-8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    parameters = tuple(module.parameters())
    if not parameters or len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError(f"{name} trainable parameter inventory is empty or duplicated")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.999),
        eps=float(adam_eps),
        weight_decay=0.0,
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError(f"{name} optimizer scope differs from its diagnostic module")
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    consecutive = 0
    early_stopped = False
    final_metrics: dict[str, Any] | None = None
    for step in range(1, int(spec["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(torch.arange(target.shape[0], device=target.device))
        if train_indices is None:
            loss = _structural_loss(target, prediction, loss_cfg)
        else:
            loss = _structural_loss(
                target.index_select(0, train_indices), prediction.index_select(0, train_indices), loss_cfg
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced non-finite loss at step {step}")
        loss.backward()
        last_code_gradient_stats = (
            None
            if code_parameter is None
            else _code_gradient_stats(
                code_parameter,
                adam_eps=float(adam_eps),
                learning_rate=float(learning_rate),
            )
        )
        optimizer.step()
        if step == 1 or step % int(spec["log_every_steps"]) == 0:
            logger.info("stage=train arm=%s step=%d/%d loss=%.8f elapsed_s=%.1f", name, step, spec["max_steps"], float(loss.detach()), time.perf_counter() - started)
        if step == 1 or step % int(spec["eval_every_steps"]) == 0 or step == int(spec["max_steps"]):
            with torch.no_grad():
                matched = predict(torch.arange(target.shape[0], device=target.device))
                swapped = predict(swap.to(target.device))
                final_metrics = _metrics(
                    target=target, matched=matched, swapped=swapped,
                    assignments=assignments, source=source, loss_cfg=loss_cfg,
                )
            if last_code_gradient_stats is not None:
                final_metrics.update(last_code_gradient_stats)
                final_metrics["code_gradient_parameter_point"] = "pre_update_at_same_step"
                final_metrics["fit_metrics_parameter_point"] = "post_update_at_same_step"
                final_metrics["optimizer_updates_completed_before_code_gradient"] = step - 1
            if train_indices is not None:
                train_set = set(int(value) for value in train_indices.tolist())
                heldout_rows = [row for row in final_metrics["tiles"] if int(row["tile_index"]) not in train_set]
                final_metrics["heldout_tile_count"] = len(heldout_rows)
                final_metrics["heldout_max_dir"] = max(row["matched_dir"] for row in heldout_rows)
                final_metrics["heldout_max_scale"] = max(row["matched_scale"] for row in heldout_rows)
                final_metrics["heldout_max_nrmse"] = max(row["matched_nrmse"] for row in heldout_rows)
            final_metrics.update({"arm": name, "step": step, "elapsed_s": time.perf_counter() - started})
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(final_metrics, sort_keys=True) + "\n")
            passed = bool(spec["early_success"]["enabled"]) and step >= int(spec["early_success"]["min_step"]) and _success(final_metrics, spec["early_success"])
            consecutive = consecutive + 1 if passed else 0
            logger.info(
                "stage=evaluate arm=%s step=%d tile_dir_max=%.6f tile_scale_max=%.6f operator_nrmse_max=%.6f swap_delta_min=%.6f pass=%s consecutive=%d",
                name, step, final_metrics["matched_max_tile_dir"], final_metrics["matched_max_tile_scale"],
                final_metrics["matched_max_operator_nrmse"], final_metrics["swap_min_operator_total_delta"],
                passed, consecutive,
            )
            if consecutive >= int(spec["early_success"]["consecutive_evals"]):
                early_stopped = True
                break
    assert final_metrics is not None
    result = {
        "schema": "weightclip_ae_progressive_bypass_arm_result_v1",
        "arm": name,
        "status": "early_success" if early_stopped else "max_steps_reached",
        "final_criteria_passed": _success(final_metrics, spec["early_success"]),
        "steps_completed": int(final_metrics["step"]),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "optimizer": {"name": "AdamW", "lr": float(learning_rate), "eps": float(adam_eps), "weight_decay": 0.0},
        "train_indices": None if train_indices is None else [int(value) for value in train_indices.tolist()],
        "final_metrics": final_metrics,
        "metrics_path": str(metrics_path),
    }
    torch.save({"arm": name, "step": result["steps_completed"], "state_dict": module.state_dict()}, output_dir / "final_trainable_state.pt")
    _atomic_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    spec = load_progressive_bypass_spec(args.config.expanduser().resolve())
    output_root = Path(str(spec["output_root"])).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"progressive-bypass output root already exists: {output_root}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("ae-two-op-progressive-bypass")
    logger.info("stage=startup checkpoint=%s device=%s dtype=bf16 seed=%s max_steps=%s output=%s", checkpoint_path, args.device, spec["seed"], spec["max_steps"], output_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    if str(cfg.model.big_vae.architecture_version) != "latent_mandatory_bridge_prenorm_v5":
        raise ValueError("progressive bypass requires an exact V5 checkpoint")
    if bool(cfg.model.big_vae.use_latent_sampling) or bool(cfg.model.big_vae.use_encoder_mu_head):
        raise ValueError("progressive bypass requires the deterministic V5 checkpoint contract")
    loss_cfg = _resolve_checkpoint_loss(cfg, spec)
    pair_cfg = cfg.train.operator_bank
    contract = {
        "schema": _SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "device": str(args.device),
        "dtype": "bf16-autocast",
        "seed": int(spec["seed"]),
        "max_steps": int(spec["max_steps"]),
        "hard_max_steps": _HARD_MAX_STEPS,
        "arms": list(_ARMS),
        "controls": ["encoder_random_z", "encoder_permuted_z"],
        "lookup_identity": "18_exact_logical_tiles_A_i_B_i",
        "loss": loss_cfg,
        "initialization_scope": {
            "direct_hidden": "plateau_checkpoint_bridge_output",
            "learnable_z": "plateau_checkpoint_encoder_z",
            "learnable_z_plus_qpos": "plateau_checkpoint_encoder_z",
            "negative_result_limit": "local_recoverability_from_plateau_basin_only",
        },
        "encoder_readout": {
            "input": "mean_over_32_actual_encoder_z_slots",
            "shared_bias_free_map": True,
            "operator_or_lineage_identity_input": False,
            "x_input": False,
            "heldout_local_tile_positions": [7, 8],
            "capacity_matched_controls": ["random_z", "permuted_z"],
        },
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
        if tuple(batch.logical_indices) != tuple(range(18)) or len(assignments) != 18 or len(set(assignments)) != 18:
            raise RuntimeError("fixed batch is not the exact 18-tile A_i/B_i cycle")
        swap = _swap_indices(assignments)
        target = batch.W.to(device)
        x = batch.x.to(device)
        x_mask = batch.x_mask.to(device)
        d_in_mask = batch.d_in_mask.to(device)
        d_out_mask = batch.d_out_mask.to(device)

        logger.info("stage=build_model checkpoint_step=%s", checkpoint["step"])
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
            initial_W_hat, _mu, _logvar, initial_dirs, debug = model.forward_debug(
                target, x, x_mask=x_mask, d_in_mask=d_in_mask, d_out_mask=d_out_mask
            )
            z_actual = debug["latent_decoder_z"].view(18, int(model.cfg.big_vae.num_latents), int(model.cfg.big_vae.d_lat)).float()
            q_base, q_pos_emb, q_pos_o, q_pos_t = model._build_decoder_query_state(
                batch_size=18, dist_patch_by_patch=debug["dist_patch_by_patch"],
                d_out=int(target.shape[2]), T=int(debug["T"]),
            )
            hidden_initial = model.mandatory_latent_bridge(
                q_base, z_actual.to(q_base.dtype), query_mask=debug["decoder_query_mask"]
            ).float()
        if not bool(d_in_mask.all()) or not bool(d_out_mask.all()) or not bool(debug["patch_mask"].all()):
            raise RuntimeError("selected diagnostic tiles must be fully valid; padded-mask equivalence is not assumed")
        with torch.no_grad():
            reconstructed_loss = _structural_loss(target, initial_W_hat, loss_cfg)
            exact_loss, _exact_details = WeightQuantileVAE.patch_structure_loss(
                target, initial_W_hat, patch_size=int(loss_cfg["patch_size"]),
                gamma=float(loss_cfg["gamma"]), lambda_dir=float(loss_cfg["lambda_dir"]),
                lambda_scale=float(loss_cfg["lambda_scale"]), lambda_rec=0.0, lambda_rel=0.0,
                huber_delta=float(loss_cfg["huber_delta"]), pred_dirs=initial_dirs,
                d_in_mask=d_in_mask, d_out_mask=d_out_mask,
            )
        parity_error = abs(float(reconstructed_loss.float().item()) - float(exact_loss.float().item()))
        if parity_error > 1.0e-5:
            raise RuntimeError(f"reconstructed-W structural loss differs from production pred_dirs path: {parity_error}")
        contract["loss_path_parity"] = {
            "all_masks_valid": True,
            "initial_absolute_error": parity_error,
            "tolerance": 1.0e-5,
        }
        _atomic_json(output_root / "resolved_contract.json", contract)

        def decode_hidden(hidden: torch.Tensor) -> torch.Tensor:
            with amp_context():
                state = hidden
                for layer in model.decoder_layers:
                    state = layer(state, q_pos=q_pos_o, q_pos2=q_pos_t, q_mask=debug["decoder_query_mask"])
                return model._decode_query_tokens_to_output(
                    state, z=z_actual.flatten(1), q_pos_emb=q_pos_emb,
                    d_in=int(target.shape[1]), d_out=int(target.shape[2]), d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]), patch_mask=debug["patch_mask"], d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask, disable_z_shortcut=True,
                )[0]

        hidden_arm = _DirectHiddenArm(hidden_initial).to(device)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("production model must remain completely frozen")
        hidden_result = _train_arm(
            name="direct_hidden", module=hidden_arm,
            predict=lambda indices, arm=hidden_arm: decode_hidden(arm.select(indices)),
            target=target, swap=swap, assignments=assignments, source=dataset.source, spec=spec,
            loss_cfg=loss_cfg, output_dir=output_root / "arm_1_direct_hidden", logger=logger,
            learning_rate=float(spec["learning_rates"]["direct_hidden"]), code_parameter=hidden_arm.hidden,
        )

        z_arm = _LearnableZArm(z_actual).to(device)

        def decode_z(arm: _LearnableZArm, indices: torch.Tensor) -> torch.Tensor:
            with amp_context():
                return model._decode_from_decoder_latent(
                    arm.select(indices), dist_patch_by_patch=debug["dist_patch_by_patch"],
                    patch_mask=debug["patch_mask"], d_in_mask=d_in_mask, d_out_mask=d_out_mask,
                    d_in=int(target.shape[1]), d_out=int(target.shape[2]), d_in_pad=int(debug["d_in_pad"]), T=int(debug["T"]),
                    disable_z_shortcut=True,
                )[0]

        z_result = _train_arm(
            name="learnable_z", module=z_arm, predict=lambda indices, arm=z_arm: decode_z(arm, indices),
            target=target, swap=swap,
            assignments=assignments, source=dataset.source, spec=spec,
            loss_cfg=loss_cfg, output_dir=output_root / "arm_2_learnable_z", logger=logger,
            learning_rate=float(spec["learning_rates"]["learnable_z"]), code_parameter=z_arm.z,
        )

        z_qpos_arm = _LearnableZArm(z_actual).to(device)

        def decode_z_plus_qpos(arm: _LearnableZArm, indices: torch.Tensor) -> torch.Tensor:
            selected = arm.select(indices)
            with amp_context():
                state = model.mandatory_latent_bridge(
                    q_base, selected.to(q_base.dtype), query_mask=debug["decoder_query_mask"]
                ) + q_pos_emb
                for layer in model.decoder_layers:
                    state = layer(state, q_pos=q_pos_o, q_pos2=q_pos_t, q_mask=debug["decoder_query_mask"])
                return model._decode_query_tokens_to_output(
                    state, z=selected.flatten(1), q_pos_emb=q_pos_emb,
                    d_in=int(target.shape[1]), d_out=int(target.shape[2]), d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]), patch_mask=debug["patch_mask"], d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask, disable_z_shortcut=True,
                )[0]

        z_qpos_result = _train_arm(
            name="learnable_z_plus_qpos", module=z_qpos_arm,
            predict=lambda indices, arm=z_qpos_arm: decode_z_plus_qpos(arm, indices),
            target=target, swap=swap, assignments=assignments, source=dataset.source, spec=spec,
            loss_cfg=loss_cfg, output_dir=output_root / "arm_3_learnable_z_plus_qpos", logger=logger,
            learning_rate=float(spec["learning_rates"]["learnable_z_plus_qpos"]), code_parameter=z_qpos_arm.z,
        )

        fixed_z = z_actual.detach()
        train_indices = torch.tensor(
            [index for index, (_key, local_index) in enumerate(assignments) if int(local_index) not in {7, 8}],
            dtype=torch.long,
            device=device,
        )
        if len(train_indices) != 14:
            raise RuntimeError("encoder readout split must hold out local tile positions 7 and 8 for both operators")
        readout = MeanLatentLinearReadout(int(model.cfg.big_vae.d_lat), tuple(target.shape[1:])).to(device)
        readout_result = _train_arm(
            name="encoder_linear_readout", module=readout,
            predict=lambda indices: readout(fixed_z.index_select(0, indices)),
            target=target, swap=swap, assignments=assignments, source=dataset.source, spec=spec,
            loss_cfg=loss_cfg, output_dir=output_root / "arm_4_encoder_linear_readout", logger=logger,
            learning_rate=float(spec["learning_rates"]["encoder_linear_readout"]), train_indices=train_indices,
        )
        random_generator = torch.Generator(device=device).manual_seed(int(spec["seed"]) + 91)
        random_z = torch.randn(fixed_z.shape, generator=random_generator, device=device, dtype=fixed_z.dtype)
        random_z = random_z * (
            fixed_z.float().square().mean().sqrt() / random_z.float().square().mean().sqrt().clamp_min(1.0e-12)
        )
        heldout_indices = torch.tensor(
            [index for index in range(18) if index not in set(int(value) for value in train_indices.tolist())],
            dtype=torch.long,
            device=device,
        )
        permutation = torch.arange(18, device=device)
        permutation[train_indices] = train_indices.flip(0)
        permutation[heldout_indices] = heldout_indices.flip(0)
        permuted_z = fixed_z.index_select(0, permutation)
        control_results = []
        for control_name, control_z in (("random_z", random_z), ("permuted_z", permuted_z)):
            control = MeanLatentLinearReadout(int(model.cfg.big_vae.d_lat), tuple(target.shape[1:])).to(device)
            control_results.append(_train_arm(
                name=f"encoder_linear_readout_{control_name}", module=control,
                predict=lambda indices, module=control, codes=control_z: module(codes.index_select(0, indices)),
                target=target, swap=swap, assignments=assignments, source=dataset.source, spec=spec,
                loss_cfg=loss_cfg, output_dir=output_root / f"control_encoder_{control_name}", logger=logger,
                learning_rate=float(spec["learning_rates"]["encoder_linear_readout"]), train_indices=train_indices,
            ))
        final_versions = tuple((name, parameter._version) for name, parameter in model.named_parameters())
        if final_versions != frozen_versions:
            raise RuntimeError("frozen production model parameter version counters changed during bypass arms")
        results = {
            "schema": "weightclip_ae_progressive_bypass_summary_v1",
            "contract": contract,
            "frozen_model_parameter_versions_unchanged": True,
            "arms": [hidden_result, z_result, z_qpos_result, readout_result],
            "encoder_readout_controls": control_results,
        }
        _atomic_json(output_root / "summary.json", results)
        logger.info("stage=complete summary=%s statuses=%s", output_root / "summary.json", [row["status"] for row in results["arms"]])


if __name__ == "__main__":
    main()
