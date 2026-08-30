from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from training.weightclip_benchmark import (
    run_gptq_token_bottleneck_comparison as baseline,
)


SCHEMA = "weightclip_conditioned_tokenizer_b_small_v1"
ARM = "conditioned_tokenizer_b"
DEFAULT_BASELINE_ROOT = Path(
    "/mnt/shared/weightclip_benchmark/"
    "raw_vs_normalized_prod_dist_3k_v1_20260827T1648Z"
)


def _parse_args() -> argparse.Namespace:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Train the minimal activation-conditioned tokenizer B in the exact "
            "known-good depth-4 normalized-weight bottleneck setup."
        )
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/shared/weightclip_benchmark/"
            f"conditioned_tokenizer_b_small_3k_v1_{stamp}"
        ),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_baseline_contract(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    required = [
        root / "COMPLETE.json",
        root / "summary.json",
        root / "config.json",
        root / "model_contract.json",
        root / "metrics.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"baseline artifact is incomplete: {missing}")
    complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("summary_sha256") != _sha256(root / "summary.json"):
        raise RuntimeError("baseline COMPLETE marker does not bind summary.json")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    model_contract = json.loads(
        (root / "model_contract.json").read_text(encoding="utf-8")
    )
    rows = _read_jsonl(root / "metrics.jsonl")
    normalized_rows = sorted(
        (row for row in rows if row.get("arm") == "normalized_float"),
        key=lambda row: int(row["step"]),
    )
    expected_steps = list(range(0, 3001, 250))
    if [int(row["step"]) for row in normalized_rows] != expected_steps:
        raise RuntimeError("baseline normalized curve is not the exact 0:250:3000 grid")
    if abs(float(normalized_rows[-1]["train"]["structural_loss"]) - 0.022273654118180275) > 1.0e-12:
        raise RuntimeError("baseline final normalized loss changed")
    return {
        "root": root,
        "config": config,
        "model_contract": model_contract,
        "normalized_rows": normalized_rows,
        "metrics_path": root / "metrics.jsonl",
    }


def _make_config(args: argparse.Namespace, baseline_contract: dict[str, Any]) -> baseline.ExperimentConfig:
    saved = baseline_contract["config"]["config"]
    cfg = baseline.ExperimentConfig(
        seed=int(saved["seed"]),
        steps=4 if args.smoke else int(args.steps),
        batch_size=2 if args.smoke else int(saved["batch_size"]),
        eval_batch_size=int(saved["eval_batch_size"]),
        eval_every=2 if args.smoke else int(args.eval_every),
        log_every=int(saved["log_every"]),
        learning_rate=float(saved["learning_rate"]),
        warmup_steps=int(saved["warmup_steps"]),
        weight_decay=float(saved["weight_decay"]),
        hidden_dim=128 if args.smoke else int(saved["hidden_dim"]),
        heads=4 if args.smoke else int(saved["heads"]),
        mlp_dim=256 if args.smoke else int(saved["mlp_dim"]),
        encoder_depth=int(saved["encoder_depth"]),
        decoder_depth=int(saved["decoder_depth"]),
        latent_slots=4 if args.smoke else int(saved["latent_slots"]),
        latent_dim=32 if args.smoke else int(saved["latent_dim"]),
        values_per_token=int(saved["values_per_token"]),
        quant_bits=int(saved["quant_bits"]),
        gptq_damp_fraction=float(saved["gptq_damp_fraction"]),
        train_activation_rows=int(saved["train_activation_rows"]),
        test_activation_rows=int(saved["test_activation_rows"]),
        structural_patch_size=int(saved["structural_patch_size"]),
        structural_gamma=float(saved["structural_gamma"]),
        structural_direction_weight=float(saved["structural_direction_weight"]),
        structural_scale_weight=float(saved["structural_scale_weight"]),
        structural_huber_delta=float(saved["structural_huber_delta"]),
        use_distribution_conditioning=True,
        distribution_k_s=int(saved["distribution_k_s"]),
        distribution_Kq=int(saved["distribution_Kq"]),
        distribution_d_var=int(saved["distribution_d_var"]),
        distribution_d_dist=int(saved["distribution_d_dist"]),
        distribution_num_var_attn_layers=int(saved["distribution_num_var_attn_layers"]),
        distribution_var_attn_heads=int(saved["distribution_var_attn_heads"]),
        distribution_dcn_num_cross_layers=int(saved["distribution_dcn_num_cross_layers"]),
        distribution_dcn_deep_hidden=int(saved["distribution_dcn_deep_hidden"]),
        distribution_dcn_deep_layers=int(saved["distribution_dcn_deep_layers"]),
        distribution_use_covariance=bool(saved["distribution_use_covariance"]),
    )
    if not args.smoke:
        exact = asdict(cfg)
        for key, value in saved.items():
            if key in exact and exact[key] != value:
                raise RuntimeError(
                    f"candidate config drifted from baseline at {key}: {exact[key]!r} != {value!r}"
                )
    return cfg


def _prepare_normalized(
    data: dict[str, Any],
    cfg: baseline.ExperimentConfig,
) -> dict[str, Any]:
    weights = data["weights"].reshape(-1, 128, 128).float()
    contexts = data["contexts"].reshape(-1, 512, 128).float()
    tile_rows = data["tile_rows"].reshape(-1)
    weight_rows = weights.transpose(1, 2).contiguous()
    qmax = float(2 ** (cfg.quant_bits - 1) - 1)
    scales = weight_rows.abs().amax(dim=-1).clamp_min(1.0e-8) / qmax
    normalized = weight_rows / scales[:, :, None]
    train_ops = set(data["train_operator_indices"])
    train_tile_mask = torch.tensor(
        [operator in train_ops for operator in range(64) for _ in range(9)],
        dtype=torch.bool,
    )
    log_scale = torch.log2(scales.clamp_min(1.0e-12))
    scale_mean = float(log_scale[train_tile_mask].mean().item())
    scale_std = float(
        log_scale[train_tile_mask].std(unbiased=False).clamp_min(1.0e-6).item()
    )
    standardized_log_scale = (log_scale - scale_mean) / scale_std
    return {
        **data,
        "flat_weights": weights,
        "flat_contexts": contexts,
        "flat_tile_rows": tile_rows,
        "normalized_tokens": normalized.reshape(-1, 128, 8, 16).reshape(
            -1, 1024, 16
        ),
        "standardized_log_scale": standardized_log_scale[:, :, None]
        .expand(-1, -1, 8)
        .reshape(-1, 1024, 1),
        "scale_log2_train_mean": scale_mean,
        "scale_log2_train_std": scale_std,
    }


class ConditionedTokenizerB(baseline.UnifiedWeightBottleneck):
    """Minimal typed tokenizer with a protected W lane and a bounded W*x lane."""

    record_width = 64

    def __init__(self, cfg: baseline.ExperimentConfig) -> None:
        if not cfg.use_distribution_conditioning:
            raise ValueError("tokenizer B requires the production Distribution Encoder")
        super().__init__(cfg, "normalized_float")
        if self.continuous_projection is None:
            raise RuntimeError("normalized baseline projection was not constructed")
        baseline_projection = self.continuous_projection.weight.detach().clone()
        del self.continuous_projection
        del self.scale_mlp
        self.tokenizer_b_activation_gate = nn.Linear(
            cfg.distribution_d_var,
            1,
            bias=False,
        )
        self.tokenizer_b_projection = nn.Linear(
            self.record_width,
            cfg.hidden_dim,
            bias=False,
        )
        nn.init.zeros_(self.tokenizer_b_activation_gate.weight)
        nn.init.normal_(self.tokenizer_b_projection.weight, std=0.02)
        with torch.no_grad():
            self.tokenizer_b_projection.weight[:, : cfg.values_per_token].copy_(
                baseline_projection
            )
            self.tokenizer_b_projection.weight[:, 34:].zero_()

    def _encode_distribution_context_full(
        self,
        activation_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.distribution_encoder is None:
            raise RuntimeError("tokenizer B requires a Distribution Encoder")
        if activation_context is None or activation_context.ndim != 3:
            raise ValueError("tokenizer B requires activation context [B,n,128]")
        batch, sample_count, d_in = activation_context.shape
        if d_in != 128:
            raise ValueError("this exact comparison requires d_in=128 activation tiles")
        patch_indices = torch.arange(
            d_in,
            device=activation_context.device,
            dtype=torch.long,
        ).view(1, 8, self.cfg.values_per_token).expand(batch, -1, -1)
        repeated_context = activation_context.unsqueeze(1).expand(
            batch, 8, sample_count, d_in
        ).reshape(batch * 8, sample_count, d_in)
        dist_var, dist_patch = self.distribution_encoder(
            repeated_context,
            patch_indices.reshape(batch * 8, self.cfg.values_per_token),
        )
        return (
            dist_var.view(
                batch,
                8,
                self.cfg.values_per_token,
                self.cfg.distribution_d_var,
            ),
            dist_patch.view(batch, 8, self.cfg.distribution_d_dist),
        )

    def build_typed_record(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        dist_var_by_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(content.shape[1:]) != (1024, self.cfg.values_per_token):
            raise ValueError(f"unexpected content shape {tuple(content.shape)}")
        if tuple(log_scale.shape) != (content.shape[0], 1024, 1):
            raise ValueError(f"unexpected log-scale shape {tuple(log_scale.shape)}")
        expected_dist = (
            content.shape[0],
            8,
            self.cfg.values_per_token,
            self.cfg.distribution_d_var,
        )
        if tuple(dist_var_by_patch.shape) != expected_dist:
            raise ValueError(
                f"unexpected per-variable context shape {tuple(dist_var_by_patch.shape)}"
            )
        content_fp32 = content.float()
        gate_patch = torch.tanh(
            self.tokenizer_b_activation_gate(dist_var_by_patch.float())
        ).squeeze(-1)
        gate = gate_patch[:, None].expand(-1, 128, -1, -1).reshape_as(content_fp32)
        interaction = content_fp32 * gate
        zero_flag = (content_fp32.abs().amax(dim=-1, keepdim=True) == 0).to(
            content_fp32.dtype
        )
        padding = content_fp32.new_zeros(content.shape[0], 1024, 30)
        record = torch.cat(
            (
                content_fp32,
                interaction,
                log_scale.float(),
                zero_flag,
                padding,
            ),
            dim=-1,
        )
        if record.shape[-1] != self.record_width:
            raise RuntimeError(f"typed record width drifted to {record.shape[-1]}")
        return record, gate

    def tokenize(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        dist_var_by_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record, gate = self.build_typed_record(content, log_scale, dist_var_by_patch)
        embedded = self.tokenizer_b_projection(record)
        group_ids = torch.arange(128, device=content.device).repeat_interleave(8)
        chunk_ids = torch.arange(8, device=content.device).repeat(128)
        token = (
            embedded
            + self.group_embedding(group_ids)[None]
            + self.chunk_embedding(chunk_ids)[None]
            + self.tile_embedding(tile_index)[:, None]
        )
        return token, record, gate

    def encode_b(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        dist_var_by_patch: torch.Tensor,
        dist_patch_by_patch: torch.Tensor,
        *,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        batch = content.shape[0]
        embedded, _record, _gate = self.tokenize(
            content,
            log_scale,
            tile_index,
            dist_var_by_patch,
        )
        latent = self.latent_slots[None].expand(batch, -1, -1)
        state = torch.cat((latent, embedded), dim=1)
        content_context = dist_patch_by_patch.unsqueeze(1).expand(
            -1, 128, -1, -1
        ).reshape(batch, 1024, self.cfg.distribution_d_dist)
        latent_context = content_context.new_zeros(
            batch,
            self.cfg.latent_slots,
            self.cfg.distribution_d_dist,
        )
        key_context = torch.cat((latent_context, content_context), dim=1)
        telemetry: list[dict[str, float]] = []
        for depth, block in enumerate(self.encoder_blocks, start=1):
            state = block(state, key_context=key_context)
            if capture_depth:
                telemetry.append(
                    {
                        "depth": depth,
                        "latent_rms": float(
                            state[:, : self.cfg.latent_slots]
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "content_rms": float(
                            state[:, self.cfg.latent_slots :]
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                    }
                )
        z = self.to_latent(self.latent_norm(state[:, : self.cfg.latent_slots]))
        return z, telemetry

    def forward(
        self,
        content: torch.Tensor,
        log_scale: torch.Tensor,
        tile_index: torch.Tensor,
        activation_context: torch.Tensor | None = None,
        *,
        capture_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        dist_var, dist_patch = self._encode_distribution_context_full(
            activation_context
        )
        z, telemetry = self.encode_b(
            content,
            log_scale,
            tile_index,
            dist_var,
            dist_patch,
            capture_depth=capture_depth,
        )
        return (
            self.decode(z, tile_index, dist_patch_by_patch=dist_patch),
            z,
            telemetry,
        )


def _shared_start_contract(
    cfg: baseline.ExperimentConfig,
    candidate: ConditionedTokenizerB,
    baseline_contract: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    baseline._seed_everything(cfg.seed)
    reference = baseline.UnifiedWeightBottleneck(cfg, "normalized_float")
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    tokenizer_prefixes = (
        "continuous_projection",
        "scale_mlp",
        "tokenizer_b_activation_gate",
        "tokenizer_b_projection",
    )
    compared: list[tuple[str, torch.Tensor]] = []
    mismatched: list[str] = []
    for name, tensor in reference_state.items():
        if name.startswith(tokenizer_prefixes):
            continue
        other = candidate_state.get(name)
        if other is None or not torch.equal(tensor, other):
            mismatched.append(name)
        else:
            compared.append((name, tensor))
    if mismatched:
        raise RuntimeError(f"candidate changed shared downstream starts: {mismatched[:8]}")
    baseline_shared = [
        (name, tensor)
        for name, tensor in reference_state.items()
        if not name.startswith(
            ("continuous_projection", "code_embedding", "code_position_gate")
        )
    ]
    baseline_digest = _tensor_digest(baseline_shared)
    expected = baseline_contract["model_contract"]["shared_start_sha256"][
        "normalized_float"
    ]
    if not smoke and baseline_digest != expected:
        raise RuntimeError(
            f"reconstructed baseline start mismatch: {baseline_digest} != {expected}"
        )
    return {
        "compared_common_tensors": len(compared),
        "common_state_sha256": _tensor_digest(compared),
        "reconstructed_baseline_shared_sha256": baseline_digest,
        "archived_baseline_shared_sha256": expected,
        "base_projection_slice_matches_reference": bool(
            torch.equal(
                candidate.tokenizer_b_projection.weight[:, : cfg.values_per_token],
                reference.continuous_projection.weight,
            )
        ),
    }


def _candidate_gradient_telemetry(model: ConditionedTokenizerB) -> dict[str, Any]:
    result = baseline._gradient_telemetry(model)
    for label, parameter in (
        ("tokenizer_b_activation_gate", model.tokenizer_b_activation_gate.weight),
        ("tokenizer_b_projection", model.tokenizer_b_projection.weight),
    ):
        if parameter.grad is None:
            result[label] = {"gradient_rms": None, "numel": parameter.numel()}
        else:
            result[label] = {
                "gradient_rms": float(
                    parameter.grad.detach().float().square().mean().sqrt().item()
                ),
                "numel": parameter.numel(),
            }
    return result


def _plot_overlay(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    base_steps = [int(row["step"]) for row in baseline_rows]
    base_loss = [float(row["train"]["structural_loss"]) for row in baseline_rows]
    cand_steps = [int(row["step"]) for row in candidate_rows]
    cand_loss = [float(row["train"]["structural_loss"]) for row in candidate_rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for axis in axes:
        axis.plot(
            base_steps,
            base_loss,
            marker="o",
            markersize=3.5,
            lw=2.0,
            label="previous normalized direct 16→512",
        )
        axis.plot(
            cand_steps,
            cand_loss,
            marker="o",
            markersize=3.5,
            lw=2.0,
            label="conditioned tokenizer B",
        )
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("train structural loss")
        axis.grid(alpha=0.25)
    axes[0].set_title("Linear loss scale")
    axes[1].set_title("Logarithmic loss scale")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Known-good depth-4 bottleneck: direct normalized input vs tokenizer B"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def _final_interventions(
    model: ConditionedTokenizerB,
    prepared: dict[str, Any],
    cfg: baseline.ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = defaultdict(float)
    for start in range(0, len(prepared["train_operator_indices"]), cfg.eval_batch_size):
        operators = prepared["train_operator_indices"][
            start : start + cfg.eval_batch_size
        ]
        batch = len(operators)
        indices = torch.tensor(
            [operator * 9 + tile for operator in operators for tile in range(9)],
            dtype=torch.long,
        )
        content = prepared["normalized_tokens"].index_select(0, indices).to(device)
        log_scale = prepared["standardized_log_scale"].index_select(0, indices).to(device)
        tile_row = prepared["flat_tile_rows"].index_select(0, indices).to(device)
        context = prepared["flat_contexts"].index_select(0, indices).to(device)
        target = prepared["flat_weights"].index_select(0, indices).to(device)
        dist_var, dist_patch = model._encode_distribution_context_full(context)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            token, record, gate = model.tokenize(
                content, log_scale, tile_row, dist_var
            )
            z, _telemetry = model.encode_b(
                content,
                log_scale,
                tile_row,
                dist_var,
                dist_patch,
            )
            matched = model.decode(z, tile_row, dist_patch_by_patch=dist_patch)
            z_by_operator = z.view(batch, 9, cfg.latent_slots, cfg.latent_dim)
            shuffled_z = z_by_operator.roll(1, dims=0).reshape_as(z)
            shuffled = model.decode(
                shuffled_z,
                tile_row,
                dist_patch_by_patch=dist_patch,
            )
            zeroed = model.decode(
                torch.zeros_like(z),
                tile_row,
                dist_patch_by_patch=dist_patch,
            )
        w_perm = torch.arange(batch).roll(1)
        flat_perm = (
            w_perm[:, None] * 9 + torch.arange(9)
        ).reshape(-1).to(device)
        content_swapped = content.index_select(0, flat_perm)
        scale_swapped = log_scale.index_select(0, flat_perm)
        context_swapped = context.view(batch, 9, 512, 128).roll(1, dims=0).reshape_as(context)
        dist_var_xswap, _dist_patch_xswap = model._encode_distribution_context_full(
            context_swapped
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            token_wswap, record_wswap, _gate_wswap = model.tokenize(
                content_swapped,
                scale_swapped,
                tile_row,
                dist_var,
            )
            token_xswap, record_xswap, _gate_xswap = model.tokenize(
                content,
                log_scale,
                tile_row,
                dist_var_xswap,
            )
        matched_loss, _ = baseline._big_vae_structural_loss(
            matched.float(), target.float(), cfg
        )
        shuffled_loss, _ = baseline._big_vae_structural_loss(
            shuffled.float(), target.float(), cfg
        )
        zeroed_loss, _ = baseline._big_vae_structural_loss(
            zeroed.float(), target.float(), cfg
        )
        count = int(target.shape[0])
        totals["count"] += count
        totals["matched_loss"] += float(matched_loss.item()) * count
        totals["shuffled_loss"] += float(shuffled_loss.item()) * count
        totals["zeroed_loss"] += float(zeroed_loss.item()) * count
        totals["matched_output_sq"] += float(matched.float().square().sum().item())
        totals["zshuffle_delta_sq"] += float(
            (shuffled.float() - matched.float()).square().sum().item()
        )
        totals["zeroz_delta_sq"] += float(
            (zeroed.float() - matched.float()).square().sum().item()
        )
        totals["token_sq"] += float(token.float().square().sum().item())
        totals["token_wswap_delta_sq"] += float(
            (token_wswap.float() - token.float()).square().sum().item()
        )
        totals["token_xswap_delta_sq"] += float(
            (token_xswap.float() - token.float()).square().sum().item()
        )
        totals["record_w_lane_max_abs_error"] = max(
            totals["record_w_lane_max_abs_error"],
            float((record[..., :16] - content.float()).abs().max().item()),
        )
        totals["record_wswap_w_lane_max_abs_error"] = max(
            totals["record_wswap_w_lane_max_abs_error"],
            float(
                (record_wswap[..., :16] - content_swapped.float())
                .abs()
                .max()
                .item()
            ),
        )
        totals["record_xswap_w_lane_max_abs_error"] = max(
            totals["record_xswap_w_lane_max_abs_error"],
            float((record_xswap[..., :16] - content.float()).abs().max().item()),
        )
        totals["gate_abs_max"] = max(
            totals["gate_abs_max"], float(gate.float().abs().max().item())
        )
    denominator_output = max(totals["matched_output_sq"], 1.0e-20)
    denominator_token = max(totals["token_sq"], 1.0e-20)
    return {
        "matched_train_structural_loss": totals["matched_loss"] / totals["count"],
        "shuffled_latent_train_structural_loss": totals["shuffled_loss"]
        / totals["count"],
        "zero_latent_train_structural_loss": totals["zeroed_loss"] / totals["count"],
        "z_shuffle_output_relative_rms_delta": math.sqrt(
            totals["zshuffle_delta_sq"] / denominator_output
        ),
        "zero_z_output_relative_rms_delta": math.sqrt(
            totals["zeroz_delta_sq"] / denominator_output
        ),
        "fixed_x_w_swap_token_relative_rms_delta": math.sqrt(
            totals["token_wswap_delta_sq"] / denominator_token
        ),
        "fixed_w_x_swap_token_relative_rms_delta": math.sqrt(
            totals["token_xswap_delta_sq"] / denominator_token
        ),
        "record_w_lane_max_abs_error": totals["record_w_lane_max_abs_error"],
        "record_wswap_w_lane_max_abs_error": totals[
            "record_wswap_w_lane_max_abs_error"
        ],
        "record_xswap_w_lane_max_abs_error": totals[
            "record_xswap_w_lane_max_abs_error"
        ],
        "gate_abs_max": totals["gate_abs_max"],
    }


def _run(args: argparse.Namespace) -> None:
    baseline_contract = _load_baseline_contract(args.baseline_root)
    cfg = _make_config(args, baseline_contract)
    if cfg.steps <= 0 or cfg.eval_every <= 0:
        raise ValueError("steps and eval interval must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise RuntimeError("the full tokenizer-B experiment requires CUDA")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    startup = {
        "schema": SCHEMA,
        "arm": ARM,
        "config": asdict(cfg),
        "device": str(device),
        "dtype": "bfloat16 autocast; fp32 structural loss",
        "seed": cfg.seed,
        "output_root": str(output_root),
        "baseline_root": str(baseline_contract["root"]),
        "baseline_metrics": str(baseline_contract["metrics_path"]),
        "selection": baseline_contract["config"]["selection"],
        "resolved_config": baseline_contract["config"]["resolved_config"],
        "cache_mode": "direct immutable mmap read; no preprocessing cache",
        "objective": baseline_contract["config"]["objective"],
        "architecture": (
            "same depth-4 unified self-attention encoder/decoder and [32,384] "
            "bottleneck as archived normalized baseline; only input tokenizer changes"
        ),
        "tokenizer_b": {
            "record": (
                "[normalized_W(16), normalized_W*tanh(shared_linear(dist_var))(16), "
                "standardized_log2_scale(1), zero_flag(1), zero_padding(30)]"
            ),
            "projection": "single bias-free Linear(64, hidden_dim)",
            "activation_conditioner": "shared bias-free Linear(256,1), tanh bounded",
            "base_weight_lane": "exact and disjoint before the single projection",
            "attention_conditioning": (
                "unchanged production Distribution Encoder patch embedding added to "
                "encoder keys and decoder queries"
            ),
        },
    }
    print("[tokenizer-b] stage=preflight", flush=True)
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[tokenizer-b] stage=dry-run-complete no_files_written=true", flush=True)
        return
    output_root.mkdir(parents=True)
    _atomic_json(output_root / "config.json", startup)
    started = time.monotonic()
    baseline._seed_everything(cfg.seed)
    print("[tokenizer-b] stage=data-load exact64=true", flush=True)
    data = baseline._load_exact64_tiles(
        Path(startup["selection"]),
        Path(startup["resolved_config"]),
    )
    prepared = _prepare_normalized(data, cfg)
    archived_summary = json.loads(
        (baseline_contract["root"] / "summary.json").read_text(encoding="utf-8")
    )
    archived_quantizer = archived_summary["quantizer_metrics"]
    if not args.smoke:
        for key, current in (
            ("scale_log2_train_mean", prepared["scale_log2_train_mean"]),
            ("scale_log2_train_std", prepared["scale_log2_train_std"]),
        ):
            if abs(float(archived_quantizer[key]) - float(current)) > 1.0e-9:
                raise RuntimeError(
                    f"normalization statistics drifted at {key}: "
                    f"{current} != {archived_quantizer[key]}"
                )
    _atomic_json(
        output_root / "data_contract.json",
        {
            "selection_sha256": data["selection_sha256"],
            "resolved_config_sha256": data["resolved_config_sha256"],
            "train_operator_indices": data["train_operator_indices"],
            "heldout_operator_indices": data["heldout_operator_indices"],
            "identities": data["identities"],
            "scale_log2_train_mean": prepared["scale_log2_train_mean"],
            "scale_log2_train_std": prepared["scale_log2_train_std"],
        },
    )
    print(
        f"[tokenizer-b] stage=model-build hidden={cfg.hidden_dim} "
        f"encoder_depth={cfg.encoder_depth} decoder_depth={cfg.decoder_depth}",
        flush=True,
    )
    baseline._seed_everything(cfg.seed)
    model = ConditionedTokenizerB(cfg)
    start_contract = _shared_start_contract(
        cfg,
        model,
        baseline_contract,
        smoke=args.smoke,
    )
    if not start_contract["base_projection_slice_matches_reference"]:
        raise RuntimeError("tokenizer B base-W projection did not inherit baseline start")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=cfg.weight_decay,
    )
    model_contract = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_tensors": sum(1 for _ in model.parameters()),
        "latent_scalars_per_tile": cfg.latent_slots * cfg.latent_dim,
        "input_scalars_per_tile": 128 * 128,
        "start_contract": start_contract,
    }
    _atomic_json(output_root / "model_contract.json", model_contract)
    baseline_rows = baseline_contract["normalized_rows"]
    train_indices = torch.tensor(data["train_operator_indices"], dtype=torch.long)
    generator = torch.Generator().manual_seed(cfg.seed + 17)
    metrics: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    metrics_path = output_root / "metrics.jsonl"
    gradient_path = output_root / "gradient_telemetry.jsonl"

    def evaluate(step: int) -> None:
        print(f"[tokenizer-b] stage=evaluate step={step}/{cfg.steps}", flush=True)
        train_metrics = baseline._evaluate_arm(
            model,
            prepared,
            data["train_operator_indices"],
            cfg,
            device,
        )
        test_metrics = baseline._evaluate_arm(
            model,
            prepared,
            data["heldout_operator_indices"],
            cfg,
            device,
        )
        row = {
            "schema": SCHEMA,
            "arm": ARM,
            "step": step,
            "train": train_metrics,
            "test_diagnostic": test_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        numeric = [
            value
            for side in (train_metrics, test_metrics)
            for value in side.values()
            if isinstance(value, (int, float))
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise RuntimeError(f"nonfinite evaluation at step {step}")
        metrics.append(row)
        _write_jsonl(metrics_path, metrics)
        _plot_overlay(
            baseline_rows,
            metrics,
            output_root / "loss_overlay_vs_previous_normalized.png",
        )
        print(
            f"[tokenizer-b] stage=evaluate step={step} "
            f"train_structural={train_metrics['structural_loss']:.6f} "
            f"train_direction={train_metrics['structural_direction_loss']:.6f} "
            f"train_scale={train_metrics['structural_scale_loss']:.6f} "
            f"latent_rank={train_metrics['latent_participation_rank']:.3f}",
            flush=True,
        )

    evaluate(0)
    print(
        f"[tokenizer-b] stage=train steps={cfg.steps} batch={cfg.batch_size} "
        f"eval_every={cfg.eval_every}",
        flush=True,
    )
    for step in range(1, cfg.steps + 1):
        warmup_scale = min(1.0, step / max(cfg.warmup_steps, 1))
        decay_progress = max(step - cfg.warmup_steps, 0) / max(
            cfg.steps - cfg.warmup_steps,
            1,
        )
        cosine_scale = 0.1 + 0.9 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
        current_lr = cfg.learning_rate * warmup_scale * cosine_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        positions = torch.randint(
            0,
            train_indices.numel(),
            (cfg.batch_size,),
            generator=generator,
        )
        sampled = train_indices.index_select(0, positions)
        cpu_indices = torch.tensor(
            [
                int(operator) * 9 + tile
                for operator in sampled.tolist()
                for tile in range(9)
            ],
            dtype=torch.long,
        )
        target = prepared["flat_weights"].index_select(0, cpu_indices).to(device)
        content = prepared["normalized_tokens"].index_select(0, cpu_indices).to(device)
        log_scale = prepared["standardized_log_scale"].index_select(0, cpu_indices).to(device)
        tile_row = prepared["flat_tile_rows"].index_select(0, cpu_indices).to(device)
        context = prepared["flat_contexts"].index_select(0, cpu_indices).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction, _z, _depth = model(
                content,
                log_scale,
                tile_row,
                context,
            )
        loss, details = baseline._big_vae_structural_loss(
            prediction.float(),
            target.float(),
            cfg,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        if step == 1 or step % cfg.eval_every == 0:
            groups = _candidate_gradient_telemetry(model)
            row = {
                "schema": SCHEMA,
                "arm": ARM,
                "step": step,
                "loss": float(loss.item()),
                "structural_direction_loss": float(details["L_dir"].item()),
                "structural_scale_loss": float(details["L_scale"].item()),
                "groups": groups,
            }
            gradients.append(row)
            _write_jsonl(gradient_path, gradients)
            if step == 1:
                if groups["none_parameter_tensors"] != 0:
                    raise RuntimeError("tokenizer-B model has parameters without gradients")
                if groups["tokenizer_b_activation_gate"]["gradient_rms"] <= 0.0:
                    raise RuntimeError("activation gate is gradient-dead at step 1")
                if groups["tokenizer_b_projection"]["gradient_rms"] <= 0.0:
                    raise RuntimeError("typed record projection is gradient-dead at step 1")
                for depth in range(1, cfg.encoder_depth + 1):
                    if groups[f"encoder_block_{depth}"]["gradient_rms"] <= 0.0:
                        raise RuntimeError(f"encoder block {depth} is gradient-dead")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % cfg.log_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"[tokenizer-b] stage=train step={step}/{cfg.steps} "
                f"loss={float(loss.item()):.6f} dir={float(details['L_dir'].item()):.6f} "
                f"scale={float(details['L_scale'].item()):.6f} lr={current_lr:.3e} "
                f"elapsed={elapsed:.1f}s rate={step / max(elapsed, 1.0e-6):.3f}_steps_per_s",
                flush=True,
            )
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(step)

    print("[tokenizer-b] stage=final-interventions", flush=True)
    interventions = _final_interventions(model, prepared, cfg, device)
    _atomic_json(output_root / "final_interventions.json", interventions)
    final_train = metrics[-1]["train"]
    baseline_final = baseline_rows[-1]["train"]
    summary = {
        "schema": SCHEMA,
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
        "executed_steps": cfg.steps,
        "candidate_final_train": final_train,
        "baseline_final_train": baseline_final,
        "candidate_minus_baseline_final_structural_loss": float(
            final_train["structural_loss"] - baseline_final["structural_loss"]
        ),
        "candidate_over_baseline_final_structural_loss": float(
            final_train["structural_loss"]
            / max(float(baseline_final["structural_loss"]), 1.0e-20)
        ),
        "interventions": interventions,
        "artifacts": {
            "config": str(output_root / "config.json"),
            "data_contract": str(output_root / "data_contract.json"),
            "model_contract": str(output_root / "model_contract.json"),
            "metrics": str(metrics_path),
            "gradient_telemetry": str(gradient_path),
            "interventions": str(output_root / "final_interventions.json"),
            "plot": str(output_root / "loss_overlay_vs_previous_normalized.png"),
            "checkpoint": str(output_root / "final_model.pt"),
        },
    }
    torch.save(model.state_dict(), output_root / "final_model.pt")
    _atomic_json(output_root / "summary.json", summary)
    _atomic_json(
        output_root / "COMPLETE.json",
        {"schema": SCHEMA, "summary_sha256": _sha256(output_root / "summary.json")},
    )
    print(
        f"[tokenizer-b] stage=complete output={output_root} "
        f"candidate_train_structural={final_train['structural_loss']:.6f} "
        f"baseline_train_structural={baseline_final['structural_loss']:.6f} "
        f"ratio={summary['candidate_over_baseline_final_structural_loss']:.3f}",
        flush=True,
    )


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
