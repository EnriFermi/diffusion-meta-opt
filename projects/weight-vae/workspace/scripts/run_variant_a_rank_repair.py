from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing import celo_meta_config, run_or_load
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
    run_dir,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    build_weight_vae,
    load_torch_cache,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
SOURCE_DATA_ROOT = ROOT / "data"
M512_BASELINE_DIR = (
    ARTIFACT_ROOT / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
DEFAULT_SUMMARY = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m1024/rank_repair_summary.csv"
)


def _slug_float(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _direction_tag(args: argparse.Namespace) -> str:
    coeff = max(0.0, float(args.block_direction_coeff))
    if coeff <= 0.0:
        return ""
    kind = str(args.block_direction_loss_kind).strip().lower()
    tag = f"_fc2dir_c{_slug_float(coeff)}"
    if kind != "mean":
        tag += f"_{kind}_f{_slug_float(float(args.block_direction_top_fraction))}"
    return tag


def _selected_lr(output_dir: Path, method: str) -> float:
    path = output_dir / "selected_lrs.csv"
    if not path.is_file():
        return float("nan")
    rows = pd.read_csv(path)
    if "selected" not in rows or "method" not in rows or "candidate_lr" not in rows:
        return float("nan")
    selected = pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int)
    sub = rows[(rows["method"].astype(str) == str(method)) & (selected == 1)]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["candidate_lr"])


def _finite_fraction(frame: pd.DataFrame, columns: list[str]) -> float:
    values = []
    for col in columns:
        if col in frame:
            values.extend(pd.to_numeric(frame[col], errors="coerce").tolist())
    if not values:
        return float("nan")
    total = len(values)
    finite = sum(math.isfinite(float(v)) for v in values)
    return float(finite / max(1, total))


def _last_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _median_numeric(frame: pd.DataFrame, column: str, *, positive_only: bool = False) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if positive_only:
        values = values[values > 0.0]
    return float(values.median()) if not values.empty else float("nan")


def _summary_for_run(
    output_dir: Path,
    *,
    role: str,
    suffix: str,
    alpha: float,
    cap: float,
    loss_clip: float,
) -> dict[str, float | int | str]:
    vae_metrics = pd.read_csv(output_dir / "vae_metrics.csv")
    diagnostics = (
        pd.read_csv(output_dir / "preconditioning_diagnostics.csv")
        if (output_dir / "preconditioning_diagnostics.csv").is_file()
        else pd.DataFrame()
    )
    downstream = pd.read_csv(output_dir / "downstream_results.csv")
    geometry = pd.read_csv(output_dir / "geometry.csv") if (output_dir / "geometry.csv").is_file() else pd.DataFrame()
    config_payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    cfg = config_payload.get("config", {})

    record_type = vae_metrics.get("record_type", pd.Series("", index=vae_metrics.index)).fillna("").astype(str)
    quality = vae_metrics[record_type == "vae_quality"].copy()
    train = vae_metrics[record_type != "vae_quality"].copy()
    eval_rows = downstream[downstream["split"].astype(str) == "eval"].copy()
    eval_dec = eval_rows[eval_rows["method"].astype(str) == "decoder_latent"]
    eval_raw = eval_rows[eval_rows["method"].astype(str) == "raw"]

    result: dict[str, float | int | str] = {
        "role": str(role),
        "suffix": str(suffix),
        "run_label": str(output_dir.name),
        "output_dir": str(output_dir),
        "config_hash": str(config_payload.get("config_hash", "")),
        "latent_dim": int(cfg.get("latent_dim", -1)),
        "vae_steps": int(cfg.get("vae_steps", -1)),
        "vae_lr": float(cfg.get("vae_lr", float("nan"))),
        "vae_batch_size": int(cfg.get("vae_batch_size", -1)),
        "eval_starts": int(cfg.get("eval_starts", -1)),
        "downstream_steps": int(cfg.get("downstream_steps", -1)),
        "geometry_eval_samples": int(cfg.get("geometry_eval_samples", -1)),
        "alpha": float(alpha),
        "cap": float(cap),
        "loss_clip": float(loss_clip),
        "decoder_eval_starts": int(eval_dec["source_weight_index"].nunique()) if "source_weight_index" in eval_dec else int(len(eval_dec)),
        "decoder_aulc_median": float(eval_dec["aulc"].median()) if not eval_dec.empty else float("nan"),
        "decoder_aulc_mean": float(eval_dec["aulc"].mean()) if not eval_dec.empty else float("nan"),
        "raw_aulc_median": float(eval_raw["aulc"].median()) if not eval_raw.empty else float("nan"),
        "raw_aulc_mean": float(eval_raw["aulc"].mean()) if not eval_raw.empty else float("nan"),
        "selected_raw_lr": _selected_lr(output_dir, "raw"),
        "selected_decoder_lr": _selected_lr(output_dir, "decoder_latent"),
        "decoded_test_loss_median": _median_numeric(quality, "decoded_test_loss"),
        "decoded_test_acc_median": _median_numeric(quality, "decoded_test_acc"),
        "raw_test_loss_median": _median_numeric(quality, "raw_test_loss"),
        "raw_test_acc_median": _median_numeric(quality, "raw_test_acc"),
        "reconstruction_rel_l2_median": _median_numeric(quality, "reconstruction_rel_l2"),
        "normalized_reconstruction_mse_median": _median_numeric(quality, "normalized_reconstruction_mse"),
        "val_recon_mse_final": _last_numeric(train, "val_recon_mse"),
        "li_A_full_per_dim_median": _median_numeric(diagnostics, "li_A_full_per_dim"),
        "li_A_full_per_dim_p95": float(pd.to_numeric(diagnostics.get("li_A_full_per_dim"), errors="coerce").quantile(0.95))
        if "li_A_full_per_dim" in diagnostics
        else float("nan"),
        "hvp_probe_a_loss_per_dim_median": _median_numeric(diagnostics, "hvp_probe_a_loss_per_dim"),
        "hvp_probe_trace_m_per_dim_median": _median_numeric(diagnostics, "hvp_probe_trace_m_per_dim"),
        "geometry_isometry_objective_median": _median_numeric(geometry, "isometry_objective"),
        "geometry_condition_median": _median_numeric(geometry, "condition_median"),
        "train_core_finite_fraction": _finite_fraction(train, ["loss", "recon_mse", "val_recon_mse"]),
        "downstream_core_finite_fraction": _finite_fraction(eval_rows, ["aulc", "final_test_loss", "best_test_loss"]),
        "block_direction_coeff": float(cfg.get("vae_block_direction_coeff", 0.0)),
        "block_direction_block": str(cfg.get("vae_block_direction_block", "")),
        "block_direction_space": str(cfg.get("vae_block_direction_space", "")),
        "block_direction_loss_kind": str(cfg.get("vae_block_direction_loss_kind", "")),
        "block_direction_top_fraction": float(cfg.get("vae_block_direction_top_fraction", 1.0)),
        "block_direction_ramp_steps": int(cfg.get("vae_block_direction_ramp_steps", 0)),
    }

    for name in [
        "train_block_recon_loss",
        "train_block_recon_effective_loss",
        "train_block_recon_rel_l2",
        "train_block_recon_grad_ratio",
        "train_precond_loss",
        "train_precond_a_loss",
        "train_precond_effective_loss",
        "train_precond_grad_scale",
        "train_precond_base_grad_norm",
        "train_precond_grad_norm",
        "train_function_anchor_loss",
        "train_function_anchor_effective_loss",
        "train_function_anchor_ce_delta",
        "train_function_anchor_margin_drop",
        "train_function_anchor_margin_drop_active_fraction",
        "train_function_anchor_grad_ratio",
        "train_block_direction_loss",
        "train_block_direction_effective_loss",
        "train_block_direction_row_cos_mean",
        "train_block_direction_row_cos_min",
        "train_block_direction_row_error_mean",
        "train_block_direction_row_error_topk_mean",
        "train_block_direction_row_error_max",
        "train_block_direction_norm_ratio_mean",
        "train_block_direction_grad_ratio",
    ]:
        positive_only = name.endswith("_grad_ratio")
        result[f"{name}_median"] = _median_numeric(train, name, positive_only=positive_only)
        result[f"{name}_final"] = _last_numeric(train, name)
    return result


def _base_overrides(
    *,
    args: argparse.Namespace,
    run_label: str,
    latent_dim: int,
    vae_steps: int,
    vae_lr: float,
    tune_starts: int,
    eval_starts: int,
    downstream_steps: int,
    geometry_eval_samples: int,
    source_dir: Path,
    init_checkpoint: str,
) -> dict[str, Any]:
    return {
        "run_label": str(run_label),
        "seed": int(args.seed),
        "device": str(args.device),
        "dtype": "float32",
        "artifact_root": str(ARTIFACT_ROOT),
        "data_root": str(SOURCE_DATA_ROOT),
        "cache_first": True,
        "force_rerun": bool(args.force_rerun),
        "show_progress": True,
        "progress_backend": "text",
        "live_vae_loss_curve": False,
        "comet_enabled": False,
        "weight_pool_source_dir": str(source_dir),
        "vae_init_checkpoint": str(init_checkpoint),
        "latent_dim": int(latent_dim),
        "vae_hidden_dim": int(args.vae_hidden_dim),
        "vae_steps": int(vae_steps),
        "vae_lr": float(vae_lr),
        "vae_batch_size": int(args.vae_batch_size),
        "geometry_eval_samples": int(geometry_eval_samples),
        "geometry_jacobian_chunk_size": int(args.geometry_jacobian_chunk_size),
        "vae_precond_diagnostic_grad_batches": int(args.precond_diagnostic_grad_batches),
        "tune_starts": int(tune_starts),
        "eval_starts": int(eval_starts),
        "downstream_steps": int(downstream_steps),
        "downstream_eval_every": int(args.downstream_eval_every),
        "downstream_batch_size": int(args.downstream_batch_size),
    }


def _baseline_cfg(args: argparse.Namespace) -> ExperimentConfig:
    latent = int(args.latent_dim)
    label = f"sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_m{latent}_rank_repair_v1_seed{int(args.seed)}"
    return celo_meta_config(
        **_base_overrides(
            args=args,
            run_label=label,
            latent_dim=latent,
            vae_steps=int(args.baseline_steps),
            vae_lr=float(args.baseline_lr),
            tune_starts=int(args.baseline_tune_starts),
            eval_starts=int(args.baseline_eval_starts),
            downstream_steps=int(args.baseline_downstream_steps),
            geometry_eval_samples=int(args.baseline_geometry_eval_samples),
            source_dir=M512_BASELINE_DIR,
            init_checkpoint="",
        ),
        vae_precond_loss_kind="none",
        vae_precond_coeff=0.0,
        vae_precond_every=0,
        vae_precond_loss_clip=0.0,
        vae_precond_max_grad_ratio=0.0,
        vae_precond_grad_clip_norm=0.0,
    )


def _finetune_cfg(args: argparse.Namespace, *, role: str, baseline_dir: Path) -> ExperimentConfig:
    latent = int(args.latent_dim)
    direction_coeff = max(0.0, float(args.block_direction_coeff))
    direction_tag = _direction_tag(args)
    if role == "control":
        label = (
            f"sage_cnn_vae_smoothing_celo_meta_control_m{latent}_fc2recon_lam0p03_"
            f"marginhuber_c0p001{direction_tag}_rank_repair_v1_seed{int(args.seed)}"
        )
        precond_kind = "none"
        alpha = 0.0
        cap = 0.0
        loss_clip = 0.0
        grad_clip = 0.0
    elif role == "a_clip20":
        label = (
            f"sage_cnn_vae_smoothing_celo_meta_li_a_hvp_m{latent}_cap0p25_clip20_"
            f"fc2recon_lam0p03_marginhuber_c0p001{direction_tag}_rank_repair_v1_seed{int(args.seed)}"
        )
        precond_kind = "li_a_hvp"
        alpha = 0.01
        cap = 0.25
        loss_clip = 20.0
        grad_clip = 1.0
    else:
        raise ValueError(f"unknown finetune role={role!r}")

    return celo_meta_config(
        **_base_overrides(
            args=args,
            run_label=label,
            latent_dim=latent,
            vae_steps=int(args.finetune_steps),
            vae_lr=float(args.finetune_lr),
            tune_starts=int(args.finetune_tune_starts),
            eval_starts=int(args.finetune_eval_starts),
            downstream_steps=int(args.finetune_downstream_steps),
            geometry_eval_samples=int(args.finetune_geometry_eval_samples),
            source_dir=baseline_dir,
            init_checkpoint=str(baseline_dir / "vae_checkpoint.pt"),
        ),
        vae_precond_loss_kind=precond_kind,
        vae_precond_coeff=alpha,
        vae_precond_burg_coeff=0.0,
        vae_precond_every=10,
        vae_precond_samples=2,
        vae_precond_pairs=4,
        vae_precond_batch_size=128,
        vae_precond_probe_scale=1.0,
        vae_precond_estimator_scope="local",
        vae_precond_hvp_mode="stopped_composite",
        vae_precond_loss_clip=loss_clip,
        vae_precond_grad_damping=0.0,
        vae_precond_warmup_steps=0,
        vae_precond_ramp_steps=1000,
        vae_precond_max_grad_ratio=cap,
        vae_precond_grad_clip_norm=grad_clip,
        vae_block_recon_coeff=0.03,
        vae_block_recon_block="fc2.weight",
        vae_block_recon_space="normalized",
        vae_block_recon_ramp_steps=1000,
        vae_block_recon_grad_diagnostic=True,
        vae_function_anchor_coeff=0.001,
        vae_function_anchor_samples=8,
        vae_function_anchor_batch_size=256,
        vae_function_anchor_loss_kind="head_margin_drop_huber",
        vae_function_anchor_block="fc2.weight",
        vae_function_anchor_ce_margin=0.0,
        vae_function_anchor_huber_delta=0.05,
        vae_function_anchor_top_fraction=1.0,
        vae_function_anchor_example_top_fraction=0.10,
        vae_function_anchor_grad_diagnostic=True,
        vae_block_direction_coeff=direction_coeff,
        vae_block_direction_block=str(args.block_direction_block),
        vae_block_direction_space=str(args.block_direction_space),
        vae_block_direction_loss_kind=str(args.block_direction_loss_kind),
        vae_block_direction_top_fraction=float(args.block_direction_top_fraction),
        vae_block_direction_ramp_steps=int(args.block_direction_ramp_steps),
        vae_block_direction_grad_diagnostic=True,
    )


def _selected_roles(only: list[str]) -> list[str]:
    roles = ["baseline", "control", "a_clip20"]
    if not only:
        return roles
    requested = [str(v).strip().lower() for v in only]
    unknown = sorted(set(requested) - set(roles))
    if unknown:
        raise ValueError(f"unknown --only roles: {unknown}; valid={roles}")
    return [role for role in roles if role in requested]


def _run_one(
    *,
    cfg: ExperimentConfig,
    role: str,
    suffix: str,
    alpha: float,
    cap: float,
    loss_clip: float,
    summary_rows: list[dict[str, float | int | str]],
    summary_out: Path,
) -> Path:
    output_dir = run_dir(cfg)
    print(
        "[variant_a_rank_repair] run "
        f"role={role} suffix={suffix} latent_dim={cfg.latent_dim} vae_steps={cfg.vae_steps} "
        f"vae_lr={cfg.vae_lr:.6g} eval_starts={cfg.eval_starts} downstream_steps={cfg.downstream_steps} "
        f"precond={cfg.vae_precond_loss_kind} alpha={alpha:.6g} cap={cap:.6g} loss_clip={loss_clip:.6g} "
        f"block_direction_coeff={float(getattr(cfg, 'vae_block_direction_coeff', 0.0)):.6g} "
        f"block_direction_block={getattr(cfg, 'vae_block_direction_block', '')} "
        f"block_direction_space={getattr(cfg, 'vae_block_direction_space', '')} "
        f"block_direction_loss_kind={getattr(cfg, 'vae_block_direction_loss_kind', '')} "
        f"block_direction_top_fraction={float(getattr(cfg, 'vae_block_direction_top_fraction', 1.0)):.6g} "
        f"block_direction_ramp_steps={int(getattr(cfg, 'vae_block_direction_ramp_steps', 0))} "
        f"output_dir={output_dir} config_hash={config_hash(cfg)}",
        flush=True,
    )
    print(f"[variant_a_rank_repair] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)
    tables = run_or_load(cfg)
    row = _summary_for_run(
        tables.output_dir,
        role=role,
        suffix=suffix,
        alpha=alpha,
        cap=cap,
        loss_clip=loss_clip,
    )
    summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    print(
        "[variant_a_rank_repair] summary "
        f"role={role} decoder_aulc_mean={row['decoder_aulc_mean']:.6g} "
        f"decoder_aulc_median={row['decoder_aulc_median']:.6g} raw_aulc_mean={row['raw_aulc_mean']:.6g} "
        f"selected_decoder_lr={row['selected_decoder_lr']:.3g} recon_rel_l2={row['reconstruction_rel_l2_median']:.6g} "
        f"decoded_test_loss={row['decoded_test_loss_median']:.6g} li_A_p95={row['li_A_full_per_dim_p95']:.6g} "
        f"trace_m={row['hvp_probe_trace_m_per_dim_median']:.6g} "
        f"dir_grad_ratio_median={row.get('train_block_direction_grad_ratio_median', float('nan')):.6g} "
        f"dir_cos_median={row.get('train_block_direction_row_cos_mean_median', float('nan')):.6g} "
        f"summary_out={summary_out}",
        flush=True,
    )
    return tables.output_dir


def _assert_cfg_guardrails(cfg: ExperimentConfig, *, role: str, expected_latent_dim: int) -> None:
    if int(cfg.latent_dim) != int(expected_latent_dim):
        raise ValueError(f"{role} latent_dim={cfg.latent_dim} expected={expected_latent_dim}")
    init_checkpoint = str(getattr(cfg, "vae_init_checkpoint", "") or "").strip()
    if role == "baseline" and init_checkpoint:
        raise ValueError(f"higher-rank baseline must be fresh, got vae_init_checkpoint={init_checkpoint}")
    if role != "baseline" and not init_checkpoint:
        raise ValueError(f"{role} finetune must warm-start from the higher-rank baseline")
    if str(cfg.weight_pool_source_dir).strip() == "":
        raise ValueError(f"{role} must use an explicit weight_pool_source_dir")
    loss_kind = str(getattr(cfg, "vae_block_direction_loss_kind", "mean")).strip().lower()
    if loss_kind not in {"mean", "topk", "worst", "tail"}:
        raise ValueError(f"{role} unsupported vae_block_direction_loss_kind={loss_kind!r}")
    top_fraction = float(getattr(cfg, "vae_block_direction_top_fraction", 1.0))
    if not math.isfinite(top_fraction) or top_fraction <= 0.0 or top_fraction > 1.0:
        raise ValueError(f"{role} vae_block_direction_top_fraction must be in (0, 1], got {top_fraction}")


def _assert_checkpoint_latent_dim(checkpoint_path: Path, cfg: ExperimentConfig, *, weight_pool_dir: Path) -> None:
    payload = load_torch_cache(checkpoint_path)
    if payload is None or not isinstance(payload.get("model_state"), dict):
        raise RuntimeError(f"could not load VAE checkpoint for latent-dim guard: {checkpoint_path}")
    weight_pool_path = weight_pool_dir / "weight_pool.pt"
    weight_payload = load_torch_cache(weight_pool_path)
    if weight_payload is None or not isinstance(weight_payload.get("weights"), torch.Tensor):
        raise RuntimeError(f"could not load weight pool for latent-dim guard: {weight_pool_path}")
    model = build_weight_vae(cfg, weight_dim=int(weight_payload["weights"].shape[1]))
    current_state = model.state_dict()
    checkpoint_state = payload["model_state"]
    if set(checkpoint_state) != set(current_state):
        missing = sorted(set(current_state) - set(checkpoint_state))[:10]
        extra = sorted(set(checkpoint_state) - set(current_state))[:10]
        raise RuntimeError(f"checkpoint key mismatch checkpoint={checkpoint_path} missing={missing} extra={extra}")
    for key, expected_tensor in current_state.items():
        checkpoint_tensor = checkpoint_state[key]
        if tuple(checkpoint_tensor.shape) != tuple(expected_tensor.shape):
            raise RuntimeError(
                f"checkpoint shape guard failed key={key} checkpoint_shape={tuple(checkpoint_tensor.shape)} "
                f"expected_shape={tuple(expected_tensor.shape)} checkpoint={checkpoint_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Variant A rank-repair experiment with a fresh higher-rank chart.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--vae-hidden-dim", type=int, default=2048)
    parser.add_argument("--vae-batch-size", type=int, default=128)
    parser.add_argument("--baseline-steps", type=int, default=150000)
    parser.add_argument("--baseline-lr", type=float, default=3e-4)
    parser.add_argument("--baseline-tune-starts", type=int, default=32)
    parser.add_argument("--baseline-eval-starts", type=int, default=64)
    parser.add_argument("--baseline-downstream-steps", type=int, default=1000)
    parser.add_argument("--baseline-geometry-eval-samples", type=int, default=16)
    parser.add_argument("--finetune-steps", type=int, default=5000)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--finetune-tune-starts", type=int, default=8)
    parser.add_argument("--finetune-eval-starts", type=int, default=64)
    parser.add_argument("--finetune-downstream-steps", type=int, default=1000)
    parser.add_argument("--finetune-geometry-eval-samples", type=int, default=16)
    parser.add_argument("--geometry-jacobian-chunk-size", type=int, default=4)
    parser.add_argument("--precond-diagnostic-grad-batches", type=int, default=32)
    parser.add_argument("--downstream-eval-every", type=int, default=25)
    parser.add_argument("--downstream-batch-size", type=int, default=128)
    parser.add_argument("--block-direction-coeff", type=float, default=0.0)
    parser.add_argument("--block-direction-block", default="fc2.weight")
    parser.add_argument("--block-direction-space", default="denormalized")
    parser.add_argument("--block-direction-loss-kind", default="mean", choices=["mean", "topk", "worst", "tail"])
    parser.add_argument("--block-direction-top-fraction", type=float, default=1.0)
    parser.add_argument("--block-direction-ramp-steps", type=int, default=1000)
    parser.add_argument("--only", action="append", default=[], help="Run only one role: baseline, control, or a_clip20.")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    roles = _selected_roles(args.only)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(
        "[variant_a_rank_repair] start "
        f"roles={roles} device={args.device} seed={args.seed} latent_dim={args.latent_dim} "
        f"block_direction_coeff={args.block_direction_coeff:.6g} "
        f"block_direction_block={args.block_direction_block} block_direction_space={args.block_direction_space} "
        f"block_direction_loss_kind={args.block_direction_loss_kind} "
        f"block_direction_top_fraction={args.block_direction_top_fraction:.6g} "
        f"block_direction_ramp_steps={args.block_direction_ramp_steps} "
        f"m512_weight_pool_source={M512_BASELINE_DIR} artifact_root={ARTIFACT_ROOT} "
        f"summary_out={args.summary_out} force_rerun={bool(args.force_rerun)}",
        flush=True,
    )

    baseline_cfg = _baseline_cfg(args)
    _assert_cfg_guardrails(baseline_cfg, role="baseline", expected_latent_dim=int(args.latent_dim))
    baseline_dir = run_dir(baseline_cfg)
    rows: list[dict[str, float | int | str]] = []

    if "baseline" in roles:
        baseline_dir = _run_one(
            cfg=baseline_cfg,
            role="baseline",
            suffix=f"baseline_m{int(args.latent_dim)}",
            alpha=0.0,
            cap=0.0,
            loss_clip=0.0,
            summary_rows=rows,
            summary_out=args.summary_out,
        )
        _assert_checkpoint_latent_dim(baseline_dir / "vae_checkpoint.pt", baseline_cfg, weight_pool_dir=baseline_dir)
    else:
        print(f"[variant_a_rank_repair] baseline skipped expected_dir={baseline_dir}", flush=True)

    if any(role in roles for role in ("control", "a_clip20")) and not (baseline_dir / "vae_checkpoint.pt").is_file():
        raise FileNotFoundError(
            "higher-rank baseline checkpoint is required before finetunes: "
            f"{baseline_dir / 'vae_checkpoint.pt'}"
        )

    for role in ["control", "a_clip20"]:
        if role not in roles:
            continue
        cfg = _finetune_cfg(args, role=role, baseline_dir=baseline_dir)
        _assert_cfg_guardrails(cfg, role=role, expected_latent_dim=int(args.latent_dim))
        _assert_checkpoint_latent_dim(Path(str(cfg.vae_init_checkpoint)), cfg, weight_pool_dir=baseline_dir)
        if role == "control":
            alpha, cap, loss_clip = 0.0, 0.0, 0.0
            dir_suffix = _direction_tag(args)
            suffix = f"control_m{int(args.latent_dim)}{dir_suffix}"
        else:
            alpha, cap, loss_clip = 0.01, 0.25, 20.0
            dir_suffix = _direction_tag(args)
            suffix = f"a_clip20_m{int(args.latent_dim)}{dir_suffix}"
        _run_one(
            cfg=cfg,
            role=role,
            suffix=suffix,
            alpha=alpha,
            cap=cap,
            loss_clip=loss_clip,
            summary_rows=rows,
            summary_out=args.summary_out,
        )

    elapsed = time.perf_counter() - t0
    print(f"[variant_a_rank_repair] done elapsed_sec={elapsed:.2f} summary_out={args.summary_out}", flush=True)


if __name__ == "__main__":
    main()
