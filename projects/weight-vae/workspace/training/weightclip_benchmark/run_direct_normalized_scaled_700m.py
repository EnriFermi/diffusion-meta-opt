from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from training.weightclip_benchmark.run_gptq_token_bottleneck_comparison import (
    DEFAULT_RESOLVED,
    DEFAULT_SELECTION,
    ExperimentConfig,
    UnifiedWeightBottleneck,
    _arm_content,
    _arm_log_scale,
    _atomic_json,
    _big_vae_structural_loss,
    _evaluate_arm,
    _gradient_telemetry,
    _load_exact64_tiles,
    _plot_metrics,
    _plot_train_loss_curves,
    _seed_everything,
)


SCHEMA = "weightclip_direct_normalized_scaled_700m_activation_metric_v2"
ARM = "normalized_float"
REFERENCE_TRAINABLE_PARAMETERS = 706_300_801
EXPECTED_PARAMETERS = 706_284_416
EXPECTED_ENCODER_PARAMETERS = 479_619_968
EXPECTED_DECODER_PARAMETERS = 226_664_448


def _parse_args() -> argparse.Namespace:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Train the single-path 706M Direct Normalized weight AE on exact64."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_RESOLVED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/shared/weightclip_benchmark/"
            f"direct_normalized_scaled_700m_p32_activation_metric_3k_v2_{stamp}"
        ),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=42,
        steps=int(args.steps),
        batch_size=6,
        eval_batch_size=2,
        eval_every=int(args.eval_every),
        log_every=int(args.log_every),
        learning_rate=1.0e-4,
        warmup_steps=320,
        weight_decay=0.01,
        hidden_dim=1536,
        heads=24,
        mlp_dim=5728,
        encoder_depth=13,
        decoder_depth=6,
        latent_slots=32,
        latent_dim=384,
        values_per_token=32,
        structural_patch_size=16,
        structural_gamma=0.5,
        structural_direction_weight=1.0,
        structural_scale_weight=0.1,
        structural_huber_delta=0.1,
        use_distribution_conditioning=True,
        distribution_k_s=64,
        distribution_Kq=128,
        distribution_d_var=256,
        distribution_d_dist=256,
        distribution_num_var_attn_layers=6,
        distribution_var_attn_heads=4,
        distribution_dcn_num_cross_layers=3,
        distribution_dcn_deep_hidden=128,
        distribution_dcn_deep_layers=3,
        distribution_use_covariance=True,
        activation_checkpointing=True,
        bounded_cosine_attention=True,
        attention_logit_scale=2.0,
        bounded_swiglu_hidden=False,
        bounded_residual_writes=False,
    )
    if args.smoke:
        cfg = replace(
            cfg,
            steps=20,
            batch_size=6,
            eval_batch_size=1,
            eval_every=10,
            log_every=1,
        )
    return cfg


def _prepare_direct_normalized(
    data: dict[str, Any], cfg: ExperimentConfig
) -> dict[str, Any]:
    if 128 % cfg.values_per_token != 0:
        raise ValueError("values_per_token must divide 128")
    chunks_per_group = 128 // cfg.values_per_token
    weights = data["weights"].reshape(-1, 128, 128).float()
    contexts = data["contexts"].reshape(-1, 512, 128).float()
    tile_rows = data["tile_rows"].reshape(-1)
    weight_rows = weights.transpose(1, 2).contiguous()
    qmax = 7.0
    scales = weight_rows.abs().amax(dim=-1).clamp_min(1.0e-8) / qmax
    normalized = weight_rows / scales[:, :, None]

    train_ops = set(data["train_operator_indices"])
    train_tile_mask = torch.tensor(
        [
            operator_index in train_ops
            for operator_index in range(64)
            for _tile in range(9)
        ],
        dtype=torch.bool,
    )
    log_scale = torch.log2(scales.clamp_min(1.0e-12))
    scale_mean = float(log_scale[train_tile_mask].mean().item())
    scale_std = float(
        log_scale[train_tile_mask].std(unbiased=False).clamp_min(1.0e-6).item()
    )
    standardized_log_scale = (log_scale - scale_mean) / scale_std
    reconstructed = normalized * scales[:, :, None]
    roundtrip_max_abs = float((reconstructed - weight_rows).abs().max().item())
    if roundtrip_max_abs > 1.0e-7:
        raise RuntimeError(
            f"Direct Normalized analytic roundtrip failed: {roundtrip_max_abs}"
        )

    token_count = 128 * chunks_per_group
    return {
        **data,
        "flat_weights": weights,
        "flat_contexts": contexts,
        "flat_tile_rows": tile_rows,
        "normalized_tokens": normalized.reshape(
            -1, 128, chunks_per_group, cfg.values_per_token
        ).reshape(-1, token_count, cfg.values_per_token),
        "standardized_log_scale": standardized_log_scale[:, :, None]
        .expand(-1, -1, chunks_per_group)
        .reshape(-1, token_count, 1),
        "normalization_contract": {
            "kind": "per-output-row symmetric maxabs/qmax, continuous values",
            "qmax": qmax,
            "chunks_per_output_row": chunks_per_group,
            "weight_tokens_per_tile": token_count,
            "log2_scale_train_mean": scale_mean,
            "log2_scale_train_std": scale_std,
            "roundtrip_max_abs": roundtrip_max_abs,
            "normalized_min": float(normalized.min().item()),
            "normalized_max": float(normalized.max().item()),
        },
    }


def _parameter_contract(model: UnifiedWeightBottleneck) -> dict[str, Any]:
    encoder_prefixes = (
        "continuous_projection",
        "scale_mlp",
        "group_embedding",
        "chunk_embedding",
        "tile_embedding",
        "distribution_encoder",
        "encoder_blocks",
        "latent_slots",
        "latent_norm",
        "to_latent",
    )
    encoder = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(encoder_prefixes)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    decoder = total - encoder
    actual = {
        "total": total,
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "encoder": encoder,
        "decoder": decoder,
        "encoder_to_decoder_ratio": encoder / decoder,
        "parameter_tensors": sum(1 for _ in model.parameters()),
    }
    expected = {
        "total": EXPECTED_PARAMETERS,
        "trainable": EXPECTED_PARAMETERS,
        "encoder": EXPECTED_ENCODER_PARAMETERS,
        "decoder": EXPECTED_DECODER_PARAMETERS,
    }
    for key, value in expected.items():
        if int(actual[key]) != value:
            raise RuntimeError(
                f"scaled model parameter contract drifted for {key}: "
                f"expected {value}, got {actual[key]}"
            )
    return actual


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _loss_panel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activation_context: torch.Tensor,
    cfg: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the sole train objective plus detached legacy diagnostics."""
    prediction_fp32 = prediction.float()
    target_fp32 = target.float()
    context_fp32 = activation_context.float()
    task_loss = BigWeightVAELossMixin.operator_relative_mse_loss(
        context_fp32,
        target_fp32,
        prediction_fp32,
    )
    with torch.no_grad():
        predicted_action = torch.matmul(context_fp32, prediction_fp32)
        target_action = torch.matmul(context_fp32, target_fp32)
        error_energy = (predicted_action - target_action).square().sum(dim=(1, 2))
        target_energy = target_action.square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        prediction_energy = predicted_action.square().sum(dim=(1, 2))
        per_tile_relative_mse = error_energy / target_energy
        action_rms_ratio = torch.sqrt(prediction_energy / target_energy)
        behavioral_operator = BigWeightVAELossMixin.operator_recon_loss(
            context_fp32,
            target_fp32,
            prediction_fp32,
        )
        behavioral_direction, behavioral_scale = (
            BigWeightVAELossMixin.operator_direction_scale_loss(
                context_fp32,
                target_fp32,
                prediction_fp32,
                gamma=0.5,
                huber_delta=0.1,
            )
        )
        structural_weighted, structural_details = _big_vae_structural_loss(
            prediction_fp32,
            target_fp32,
            cfg,
        )
        structural_direction = structural_details["L_dir"]
        structural_scale = structural_details["L_scale"]
        behavioral = behavioral_direction + behavioral_scale
        structural_equal_sum = structural_direction + structural_scale
        legacy_balanced_nominal = behavioral + structural_equal_sum
    return task_loss, {
        "activation_relative_mse": task_loss.detach(),
        "activation_relative_mse_median": per_tile_relative_mse.median().detach(),
        "activation_relative_mse_p95": torch.quantile(
            per_tile_relative_mse, 0.95
        ).detach(),
        "activation_relative_mse_max": per_tile_relative_mse.max().detach(),
        "activation_pred_target_rms_ratio_median": action_rms_ratio.median().detach(),
        "activation_pred_target_rms_ratio_p95": torch.quantile(
            action_rms_ratio, 0.95
        ).detach(),
        "activation_pred_target_rms_ratio_max": action_rms_ratio.max().detach(),
        "behavioral": behavioral.detach(),
        "behavioral_operator": behavioral_operator.detach(),
        "behavioral_direction": behavioral_direction.detach(),
        "behavioral_scale": behavioral_scale.detach(),
        "structural": structural_equal_sum.detach(),
        "structural_direction": structural_direction.detach(),
        "structural_scale": structural_scale.detach(),
        "structural_direction_plus_0p1_scale": structural_weighted.detach(),
        "legacy_balanced_nominal_loss": legacy_balanced_nominal.detach(),
    }


def _plot_all_loss_curves(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    keys = (
        ("activation_relative_mse", "actual activation-relative MSE"),
        ("behavioral_direction", "behavioral direction"),
        ("behavioral_scale", "behavioral scale"),
        ("structural_direction", "structural direction"),
        ("structural_scale", "structural scale"),
        ("legacy_balanced_nominal_loss", "legacy nominal total"),
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )
    steps = [int(row["step"]) for row in rows]
    top_keys = keys[:1] + keys[-1:]
    component_keys = keys[1:-1]
    for key, label in top_keys:
        axes[0].plot(
            steps,
            [float(row[key]) for row in rows],
            label=label,
            linewidth=1.4,
        )
    for key, label in component_keys:
        axes[1].plot(
            steps,
            [float(row[key]) for row in rows],
            label=label,
            linewidth=1.4,
        )
    axes[0].set_ylabel("objective / nominal total")
    axes[0].set_title("Direct Normalized 700M — actual objective and legacy diagnostics")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("detached loss component")
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def _evaluate_loss_panel(
    model: UnifiedWeightBottleneck,
    prepared: dict[str, Any],
    operator_indices: list[int],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for start in range(0, len(operator_indices), cfg.eval_batch_size):
        operator_batch = operator_indices[start : start + cfg.eval_batch_size]
        cpu_indices = torch.tensor(
            [operator * 9 + tile for operator in operator_batch for tile in range(9)],
            dtype=torch.long,
        )
        target = prepared["flat_weights"].index_select(0, cpu_indices).to(device)
        tile_row = prepared["flat_tile_rows"].index_select(0, cpu_indices).to(device)
        activation_context = prepared["flat_contexts"].index_select(0, cpu_indices).to(
            device
        )
        content = _arm_content(prepared, ARM, cpu_indices).to(device)
        log_scale = _arm_log_scale(prepared, ARM, cpu_indices).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            prediction, _latent, _depth = model(
                content,
                log_scale,
                tile_row,
                activation_context,
            )
        _actual, parts = _loss_panel(
            prediction,
            target,
            activation_context,
            cfg,
        )
        tiles = int(cpu_indices.numel())
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * tiles
        count += tiles
    if count != len(operator_indices) * 9:
        raise RuntimeError("loss-panel evaluation did not cover every operator tile")
    return {key: value / count for key, value in sums.items()}


def _run(args: argparse.Namespace) -> None:
    cfg = _config(args)
    if cfg.steps <= 0 or cfg.eval_every <= 0 or cfg.log_every <= 0:
        raise ValueError("steps/eval/log intervals must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.dry_run and device.type != "cuda":
        raise RuntimeError("the scaled experiment requires CUDA")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root already exists: {output_root}")

    physical_operators = 2
    accumulation_steps = 3
    startup = {
        "schema": SCHEMA,
        "config": asdict(cfg),
        "arm": ARM,
        "device": str(device),
        "dtype": "BF16 autocast matmuls; FP32 normalization and structural loss",
        "seed": cfg.seed,
        "output_root": str(output_root),
        "selection": str(args.selection.expanduser().resolve()),
        "resolved_config": str(args.resolved_config.expanduser().resolve()),
        "cache_mode": "immutable exact64 mmap input; direct in-memory normalization",
        "logical_batch": {
            "operators_per_update": cfg.batch_size,
            "physical_operators_per_microbatch": physical_operators,
            "gradient_accumulation_steps": accumulation_steps,
            "tiles_per_update": cfg.batch_size * 9,
        },
        "architecture": {
            "path": "single encoder -> [32,384] -> single decoder",
            "input_projection": "32 continuous normalized weights -> 1536",
            "weight_tokens_per_tile": 512,
            "encoder_sequence_length": 544,
            "activation_conditioning": (
                "production Distribution Encoder; four aligned 32-variable contexts; "
                "added to encoder keys and decoder queries only"
            ),
            "no_bypass_or_auxiliary_path": True,
            "attention": "per-head L2-normalized Q/K, fixed cosine scale 2.0",
            "residual": (
                "ordinary pre-norm residual writes; depth-scaled output initialization; "
                "no SwiGLU or residual RMS caps"
            ),
        },
        "objective": (
            "sole backward objective = per-tile mean(||X@(W_hat-W)||^2 / "
            "(||X@W||^2+eps)); behavioral operator/direction/scale and structural "
            "direction/scale are detached diagnostics only"
        ),
    }
    print("[direct-normalized-700m] stage=preflight", flush=True)
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        with torch.device("meta"):
            meta_model = UnifiedWeightBottleneck(cfg, ARM)
        startup["parameter_contract"] = _parameter_contract(meta_model)
        print(
            json.dumps(startup["parameter_contract"], indent=2, sort_keys=True),
            flush=True,
        )
        print(
            "[direct-normalized-700m] stage=dry-run-complete no_files_written=true",
            flush=True,
        )
        return

    output_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_root / "config.json", startup)
    started = time.monotonic()
    print("[direct-normalized-700m] stage=data-load", flush=True)
    _seed_everything(cfg.seed)
    data = _load_exact64_tiles(args.selection.resolve(), args.resolved_config.resolve())
    prepared = _prepare_direct_normalized(data, cfg)
    _atomic_json(
        output_root / "data_contract.json",
        {
            "selection_sha256": prepared["selection_sha256"],
            "resolved_config_sha256": prepared["resolved_config_sha256"],
            "pair_manifest": prepared["pair_manifest"],
            "identities": prepared["identities"],
            "train_operator_indices": prepared["train_operator_indices"],
            "heldout_operator_indices": prepared["heldout_operator_indices"],
            "normalization": prepared["normalization_contract"],
        },
    )

    print("[direct-normalized-700m] stage=model-build", flush=True)
    _seed_everything(cfg.seed)
    model = UnifiedWeightBottleneck(cfg, ARM).to(device)
    parameter_contract = _parameter_contract(model)
    startup["parameter_contract"] = parameter_contract
    _atomic_json(output_root / "config.json", startup)
    print(
        "[direct-normalized-700m] stage=model-build-complete "
        f"parameters={parameter_contract['total']} "
        f"encoder={parameter_contract['encoder']} "
        f"decoder={parameter_contract['decoder']}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=cfg.weight_decay,
    )
    torch.cuda.reset_peak_memory_stats(device)

    train_operator_indices = torch.tensor(
        prepared["train_operator_indices"], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(cfg.seed + 17)
    eval_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    metrics_path = output_root / "metrics.jsonl"
    grad_path = output_root / "gradient_telemetry.jsonl"
    train_metrics_path = output_root / "train_metrics.jsonl"

    def evaluate(step: int) -> None:
        print(
            f"[direct-normalized-700m] stage=evaluate step={step}/{cfg.steps}",
            flush=True,
        )
        train_eval_indices = (
            prepared["train_operator_indices"][:2]
            if args.smoke
            else prepared["train_operator_indices"]
        )
        diagnostic_eval_indices = (
            prepared["heldout_operator_indices"][:1]
            if args.smoke
            else prepared["heldout_operator_indices"]
        )
        train_metrics = _evaluate_arm(
            model,
            prepared,
            train_eval_indices,
            cfg,
            device,
        )
        diagnostic_metrics = _evaluate_arm(
            model,
            prepared,
            diagnostic_eval_indices,
            cfg,
            device,
        )
        train_metrics.update(
            _evaluate_loss_panel(
                model,
                prepared,
                train_eval_indices,
                cfg,
                device,
            )
        )
        row = {
            "schema": SCHEMA,
            "step": step,
            "arm": ARM,
            "train": train_metrics,
            "test": diagnostic_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        for side in (train_metrics, diagnostic_metrics):
            for key, value in side.items():
                if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                    raise RuntimeError(
                        f"nonfinite evaluation metric at step {step}: {key}={value}"
                    )
        eval_rows.append(row)
        _write_jsonl(metrics_path, eval_rows)
        _plot_metrics(eval_rows, output_root / "training_metrics.png")
        _plot_train_loss_curves(eval_rows, output_root / "train_loss_curve.png")
        _plot_all_loss_curves(
            [{"step": item["step"], **item["train"]} for item in eval_rows],
            output_root / "all_train_loss_curves.png",
        )
        print(
            f"[direct-normalized-700m] stage=evaluate-complete step={step} "
            f"task_loss={train_metrics['activation_relative_mse']:.6f} "
            f"behavioral_dir={train_metrics['behavioral_direction']:.6f} "
            f"behavioral_scale={train_metrics['behavioral_scale']:.6f} "
            f"structural_dir={train_metrics['structural_direction']:.6f} "
            f"structural_scale={train_metrics['structural_scale']:.6f}",
            flush=True,
        )

    evaluate(0)
    print(
        f"[direct-normalized-700m] stage=train steps={cfg.steps} "
        f"effective_operators={cfg.batch_size} micro_operators={physical_operators} "
        f"accum={accumulation_steps}",
        flush=True,
    )
    if cfg.batch_size != physical_operators * accumulation_steps:
        raise RuntimeError("logical batch contract drifted")

    for step in range(1, cfg.steps + 1):
        warmup_scale = min(1.0, step / max(cfg.warmup_steps, 1))
        decay_progress = max(step - cfg.warmup_steps, 0) / max(
            cfg.steps - cfg.warmup_steps, 1
        )
        cosine_scale = 0.1 + 0.9 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
        current_lr = cfg.learning_rate * warmup_scale * cosine_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        sample_positions = torch.randint(
            0,
            train_operator_indices.numel(),
            (cfg.batch_size,),
            generator=generator,
        )
        sampled_operators = train_operator_indices.index_select(0, sample_positions)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_parts: dict[str, float] = {}
        for micro_index in range(accumulation_steps):
            micro_operators = sampled_operators[
                micro_index * physical_operators : (micro_index + 1) * physical_operators
            ]
            cpu_indices = torch.tensor(
                [
                    int(operator) * 9 + tile
                    for operator in micro_operators.tolist()
                    for tile in range(9)
                ],
                dtype=torch.long,
            )
            target = prepared["flat_weights"].index_select(0, cpu_indices).to(device)
            tile_row = prepared["flat_tile_rows"].index_select(0, cpu_indices).to(device)
            activation_context = prepared["flat_contexts"].index_select(
                0, cpu_indices
            ).to(device)
            content = _arm_content(prepared, ARM, cpu_indices).to(device)
            log_scale = _arm_log_scale(prepared, ARM, cpu_indices).to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                prediction, _latent, _depth = model(
                    content,
                    log_scale,
                    tile_row,
                    activation_context,
                )
            micro_loss, parts = _loss_panel(
                prediction,
                target,
                activation_context,
                cfg,
            )
            if not bool(torch.isfinite(micro_loss)):
                raise RuntimeError(f"nonfinite loss at step {step} micro {micro_index}")
            (micro_loss / accumulation_steps).backward()
            step_loss += float(micro_loss.item()) / accumulation_steps
            for key, value in parts.items():
                step_parts[key] = step_parts.get(key, 0.0) + float(value.item()) / accumulation_steps

        if step == 1 or step % cfg.eval_every == 0:
            telemetry = {
                "schema": SCHEMA,
                "step": step,
                "arm": ARM,
                "loss": step_loss,
                **step_parts,
                "groups": _gradient_telemetry(model),
            }
            gradient_rows.append(telemetry)
            _write_jsonl(grad_path, gradient_rows)
        if step == 1:
            groups = gradient_rows[-1]["groups"]
            if groups["none_parameter_tensors"] != 0:
                raise RuntimeError("scaled model has parameters without gradients at step 1")
            for depth in range(1, cfg.encoder_depth + 1):
                if groups[f"encoder_block_{depth}"]["gradient_rms"] <= 0.0:
                    raise RuntimeError(f"encoder block {depth} is gradient-dead")
            for depth in range(1, cfg.decoder_depth + 1):
                if groups[f"decoder_block_{depth}"]["gradient_rms"] <= 0.0:
                    raise RuntimeError(f"decoder block {depth} is gradient-dead")
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        )
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite global gradient norm at step {step}")
        optimizer.step()

        if step == 1 or step % cfg.log_every == 0:
            elapsed = time.monotonic() - started
            allocated_gib = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gib = torch.cuda.memory_reserved(device) / (1024**3)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(
                f"[direct-normalized-700m] stage=train step={step}/{cfg.steps} "
                f"loss={step_loss:.6f} "
                f"behavioral_dir={step_parts['behavioral_direction']:.6f} "
                f"behavioral_scale={step_parts['behavioral_scale']:.6f} "
                f"structural_dir={step_parts['structural_direction']:.6f} "
                f"structural_scale={step_parts['structural_scale']:.6f} "
                f"grad_norm={grad_norm:.4e} lr={current_lr:.3e} "
                f"elapsed={elapsed:.1f}s rate={step/max(elapsed, 1.0e-6):.3f}_steps_per_s",
                f"cuda_allocated={allocated_gib:.2f}GiB "
                f"cuda_reserved={reserved_gib:.2f}GiB cuda_peak={peak_gib:.2f}GiB",
                flush=True,
            )
            train_row = {
                "schema": SCHEMA,
                "step": step,
                "loss": step_loss,
                **step_parts,
                "grad_norm_pre_clip": grad_norm,
                "learning_rate": current_lr,
                "elapsed_seconds": elapsed,
                "steps_per_second": step / max(elapsed, 1.0e-6),
                "cuda_peak_gib": peak_gib,
            }
            for key, value in train_row.items():
                if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                    raise RuntimeError(f"nonfinite train telemetry {key}={value} at step {step}")
            train_rows.append(train_row)
            _write_jsonl(train_metrics_path, train_rows)
            _plot_all_loss_curves(train_rows, output_root / "all_train_loss_curves_live.png")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(step)

    checkpoint_path: Path | None = None
    if not args.smoke:
        print("[direct-normalized-700m] stage=checkpoint-write", flush=True)
        checkpoint_path = output_root / f"model_step_{cfg.steps:07d}.pt"
        checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": SCHEMA,
                "step": cfg.steps,
                "config": asdict(cfg),
                "model_state": model.state_dict(),
            },
            checkpoint_tmp,
        )
        checkpoint_tmp.replace(checkpoint_path)
    final_train = eval_rows[-1]["train"]
    if args.smoke and not (
        float(final_train["activation_relative_mse"])
        < float(eval_rows[0]["train"]["activation_relative_mse"])
    ):
        raise RuntimeError(
            "representative smoke failed to improve the sole activation-space objective"
        )
    summary = {
        "schema": SCHEMA,
        "complete": True,
        "steps": cfg.steps,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_contract": parameter_contract,
        "normalization_contract": prepared["normalization_contract"],
        "final_train": final_train,
        "final_diagnostic": eval_rows[-1]["test"],
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "cuda_memory": {
            "allocated_gib_at_end": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gib_at_end": torch.cuda.memory_reserved(device) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "artifacts": {
            "metrics": str(metrics_path),
            "gradients": str(grad_path),
            "plot": str(output_root / "train_loss_curve.png"),
            "train_metrics": str(train_metrics_path),
            "all_train_losses_plot": str(output_root / "all_train_loss_curves.png"),
            "all_train_losses_live_plot": str(
                output_root / "all_train_loss_curves_live.png"
            ),
        },
    }
    _atomic_json(output_root / "summary.json", summary)
    _atomic_json(
        output_root / "COMPLETE.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "summary": str(output_root / "summary.json"),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        },
    )
    print(
        f"[direct-normalized-700m] stage=complete output={output_root} "
        f"train_loss={final_train['activation_relative_mse']:.6f} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
