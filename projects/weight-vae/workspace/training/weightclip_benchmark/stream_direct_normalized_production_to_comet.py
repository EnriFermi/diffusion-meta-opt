from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

from comet_ml import Experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a live direct-normalized production run to Comet.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project", default="big_weight_vae")
    parser.add_argument("--name", default="direct-normalized-p32-700m-production-500k-v1")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gradient_row_metrics(
    row: dict[str, Any], *, raw_direction_mse: bool
) -> dict[str, float]:
    """Flatten legacy and mini-preference gradient rows without schema guesses."""

    metrics: dict[str, float] = {}
    structured = {
        "groups",
        "model_groups",
        "projector_groups",
        "objective_gradient_groups",
        "direction_scale_component_gradients",
        "latent_objective_gradients",
        "latent_code_scale_gradients",
    }
    for name, value in row.items():
        if name in {"schema", "step", *structured}:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"gradient/{name}"] = float(value)
    for namespace, groups in (
        ("", row.get("groups", {})),
        ("model/", row.get("model_groups", {})),
        ("projector/", row.get("projector_groups", {})),
    ):
        for group, payload in groups.items():
            if isinstance(payload, dict) and "gradient_rms" in payload:
                metrics[f"grad_rms/{namespace}{group}"] = float(
                    payload["gradient_rms"]
                )
    for objective, groups in row.get("objective_gradient_groups", {}).items():
        if not isinstance(groups, dict):
            continue
        for group, payload in groups.items():
            if not isinstance(payload, dict):
                continue
            for metric_name in ("gradient_l2", "gradient_rms"):
                value = payload.get(metric_name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[
                        f"objective_gradient/{objective}/{group}/{metric_name}"
                    ] = float(value)
    for name, value in row.get("latent_objective_gradients", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"latent_gradient/{name}"] = float(value)
    for name, value in row.get("latent_code_scale_gradients", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"categorical_latent_gradient/{name}"] = float(value)
    for parameter, payload in row.get(
        "direction_scale_component_gradients", {}
    ).items():
        if not isinstance(payload, dict):
            continue
        for name, value in payload.items():
            if (
                not raw_direction_mse
                and name.startswith("direction_contrastive")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                metrics[
                    f"direction_contrastive_gradient/{parameter}/{name}"
                ] = float(value)
    return metrics


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.resolve()
    metrics_path = run_root / "train_metrics.jsonl"
    gradients_path = run_root / "gradient_telemetry.jsonl"
    probe_path = run_root / "fixed_probe_metrics.jsonl"
    resolved_path = run_root / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    workspace = os.environ.get("COMET_WORKSPACE", "").strip()
    if not os.environ.get("COMET_API_KEY", "").strip() or not workspace:
        raise RuntimeError("COMET_API_KEY and COMET_WORKSPACE must be sourced before launch")

    experiment = Experiment(
        project_name=args.project,
        workspace=workspace,
        auto_output_logging=None,
        auto_metric_logging=False,
        auto_param_logging=False,
        log_code=False,
    )
    resolved = json.loads(resolved_path.read_text())
    config = resolved["resolved_config"]
    model = resolved["model_config"]
    schema = str(resolved["schema"])
    realized_steps = int(resolved.get("steps", config["steps"]))
    scientific_horizon_steps = int(
        resolved.get("scientific_horizon_steps", config["steps"])
    )
    smoke = realized_steps < scientific_horizon_steps
    parameter_training_scope = resolved.get("parameter_training_scope", {})
    strict_encoder_only = parameter_training_scope.get("kind") == (
        "strict_encoder_only_via_z"
    )
    decoder_thaw = schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "polar_tails_encoder_warm_decoder_lr_ramp_production_v1"
    )
    categorical_gptq = schema == (
        "weightclip_parallel_categorical_gptq_700m_production_v1"
    )
    mini_latent_preference = schema == (
        "weightclip_mini_polar_latent_preference_production_v1"
    )
    mini_polar = mini_latent_preference or schema == (
        "weightclip_mini_polar_regression_production_v1"
    )
    latent_rooted = schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "latent_rooted_polar_tails_production_v1"
    )
    raw_direction_mse_two_loss = schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "polar_tails_raw_direction_mse_structural_scale_production_v1"
    )
    raw_direction_mse = raw_direction_mse_two_loss or schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "polar_tails_raw_direction_mse_production_v1"
    )
    direction_gauge_fixed = schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "polar_tails_latent_anticollapse_direction_infonce_"
        "gauge_fixed_production_v1"
    )
    direction_infonce = direction_gauge_fixed or schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "polar_tails_latent_anticollapse_direction_infonce_production_v1"
    )
    latent_anticollapse = direction_infonce or schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "polar_tails_latent_anticollapse_production_v1"
    )
    is_polar = (
        mini_polar
        or decoder_thaw
        or raw_direction_mse
        or latent_rooted
        or latent_anticollapse
        or schema
        == "weightclip_direct_normalized_scaled_700m_polar_tails_production_v1"
    )
    default_parameter_count = (
        9_938_689
        if mini_polar
        else (
            696_787_328
            if categorical_gptq
            else (
                742_119_297
                if direction_gauge_fixed
                else (742_120_833 if is_polar else 706_301_312)
            )
        )
    )
    parameter_count = int(
        resolved.get("total_parameter_count", default_parameter_count)
    )
    architecture = (
        "mini_conditioned_p16_tile32_polar_tails"
        if mini_polar
        else (
            "direct_normalized_p32_parallel_p16_categorical_gptq"
            if categorical_gptq
            else (
                "direct_normalized_transformer_p32_latent_rooted_polar_tails"
                if latent_rooted
                else (
                    "direct_normalized_transformer_p32_polar_tails_gauge_fixed_direction_head"
                    if direction_gauge_fixed
                    else (
                        "direct_normalized_transformer_p32_polar_tails"
                        if is_polar
                        else "direct_normalized_transformer_p32"
                    )
                )
            )
        )
    )
    tags = (
        ["mini", "10m", "tile32", "p16", "distribution-encoder", "regression"]
        if mini_polar
        else ["direct-normalized", "p32", "700m"]
    )
    tags.extend(["smoke"] if smoke else ["production", "500k"])
    if is_polar:
        tags.extend(["polar-tails", "coordinate-owned-gradient-routing"])
    if mini_latent_preference:
        tags.extend(
            [
                "matched-hard-negatives",
                "latent-representation-infonce",
                "decoder-latent-preference",
                "canonical-pairs",
            ]
        )
    if strict_encoder_only:
        tags.extend(
            [
                "encoder-only",
                "decoder-frozen",
                "direct-decoder-conditioning-frozen",
            ]
        )
    if decoder_thaw:
        tags.extend(
            [
                "encoder-warmstart",
                "decoder-thaw",
                "decoder-linear-lr-ramp-10k",
                "encoder-adam-preserved",
                "decoder-adam-fresh",
            ]
        )
    if latent_rooted:
        tags.extend(["latent-rooted-decoder", "no-query-value-bypass"])
    if latent_anticollapse:
        tags.extend(["latent-anticollapse", "two-way-centered-rms-hinge"])
    if direction_infonce:
        tags.extend(
            [
                "direction-infonce",
                "same-layout-negatives",
                "target-detached",
                "equal-group-mean",
            ]
        )
    if raw_direction_mse:
        tags.extend(
            [
                "raw-direction-mse",
                "direction-k1",
                "cosine-loss-disabled",
                "infonce-disabled",
                "radius-supervised",
            ]
        )
    if raw_direction_mse_two_loss:
        tags.extend(
            [
                "two-loss-objective",
                "structural-scale-only",
                "behavioral-scale-disabled",
            ]
        )
    if direction_gauge_fixed:
        tags.extend(
            [
                "direction-gauge-fixed",
                "nonaffine-direction-rmsnorm",
                "frobenius-sphere",
                "tangent-first-moment",
            ]
        )
    if categorical_gptq:
        tags.extend(
            [
                "categorical-gptq",
                "parallel-p16",
                "ordinal-cumulative-loss",
                "categorical-scale",
                "decoder-cross-attention-to-z",
                "old-losses-detached",
            ]
        )
    experiment.set_name(args.name)
    experiment.add_tags(tags)
    experiment.log_parameters(
        {
            "architecture": architecture,
            "parameters": parameter_count,
            "steps": realized_steps,
            "scientific_horizon_steps": scientific_horizon_steps,
            "smoke": smoke,
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "scheduler": config["scheduler"],
            "weight_decay": config["weight_decay"],
            "grad_clip_norm": config["grad_clip_norm"],
            "encoder_depth": model["encoder_depth"],
            "decoder_depth": model.get(
                "decoder_depth", model.get("shared_decoder_depth")
            ),
            "hidden_dim": model["hidden_dim"],
            "latent_shape": f"{model['latent_slots']}x{model['latent_dim']}",
            "values_per_token": model["values_per_token"],
            "run_root": str(run_root),
            **(
                {
                    "tile_size": model["tile_size"],
                    "output_patch_size": model["output_patch_size"],
                    "shared_decoder_depth": model["shared_decoder_depth"],
                    "direction_tail_depth": model["direction_tail_depth"],
                    "scale_tail_depth": model["scale_tail_depth"],
                    "distribution_k_s": model["distribution_k_s"],
                    "distribution_d_var": model["distribution_d_var"],
                    "distribution_d_dist": model["distribution_d_dist"],
                    "distribution_conditioning": model[
                        "use_distribution_conditioning"
                    ],
                    "normalization_stats_path": config["normalization"][
                        "stats_path"
                    ],
                }
                if mini_polar
                else {}
            ),
            **(
                {
                    "pairing_kind": config["pairing"]["kind"],
                    "pairs_per_batch": config["pairing"]["pairs_per_batch"],
                    "representation_kind": config["representation"]["kind"],
                    "representation_temperature": config["representation"]["temperature"],
                    "preference_kind": config["preference"]["kind"],
                    "preference_margin": config["preference"]["margin"],
                    "preference_temperature": config["preference"]["temperature"],
                    "auxiliary_ramp_steps": config["calibration"]["ramp_steps"],
                }
                if mini_latent_preference
                else {}
            ),
            **(
                {
                    "training_scope": parameter_training_scope["kind"],
                    "trainable_parameters": parameter_training_scope[
                        "trainable_parameters"
                    ],
                    "frozen_parameters": parameter_training_scope[
                        "frozen_parameters"
                    ],
                    "decoder_frozen": parameter_training_scope["decoder_frozen"],
                    "direct_decoder_conditioning_frozen": parameter_training_scope[
                        "direct_decoder_conditioning_frozen"
                    ],
                }
                if strict_encoder_only
                else {}
            ),
            **(
                {
                    "decoder_thaw_kind": config["decoder_thaw"]["kind"],
                    "decoder_thaw_source_checkpoint": config["decoder_thaw"][
                        "source_checkpoint"
                    ],
                    "decoder_thaw_source_step": config["decoder_thaw"]["source_step"],
                    "decoder_thaw_source_logical_index": config["decoder_thaw"][
                        "source_committed_logical_index"
                    ],
                    "encoder_learning_rate": config["decoder_thaw"][
                        "encoder_learning_rate"
                    ],
                    "decoder_start_learning_rate": config["decoder_thaw"][
                        "decoder_start_learning_rate"
                    ],
                    "decoder_target_learning_rate": config["decoder_thaw"][
                        "decoder_target_learning_rate"
                    ],
                    "decoder_lr_ramp_steps": config["decoder_thaw"]["ramp_steps"],
                }
                if decoder_thaw
                else {}
            ),
            **(
                {
                    "categorical_code_vocab_size": config["architecture"][
                        "code_vocab_size"
                    ],
                    "categorical_scale_vocab_size": config["architecture"][
                        "scale_vocab_size"
                    ],
                    "categorical_decoder_contract": config["architecture"][
                        "decoder_contract"
                    ],
                    "gptq_kind": config["gptq"]["kind"],
                    "gptq_bits": config["gptq"]["bits"],
                    "gptq_damp_fraction": config["gptq"]["damp_fraction"],
                    "scale_log2_min": config["scale_bins"]["log2_min"],
                    "scale_log2_max": config["scale_bins"]["log2_max"],
                    "categorical_loss_kind": config["loss"]["kind"],
                    "old_losses_detached": config["diagnostics"][
                        "old_losses_detached"
                    ],
                }
                if categorical_gptq
                else {}
            ),
            **(
                {
                    "direction_regression_kind": config["direction_regression"]["kind"],
                    "direction_regression_components": config["direction_regression"][
                        "components"
                    ],
                    "direction_regression_target_radius": config[
                        "direction_regression"
                    ]["target_radius"],
                    "direction_regression_normalization": config[
                        "direction_regression"
                    ]["normalization"],
                    "direction_regression_patch_weighting": config[
                        "direction_regression"
                    ]["patch_weighting"],
                    "direction_cosine_loss_enabled": False,
                    "direction_infonce_enabled": False,
                    "behavioral_scale_enabled": not raw_direction_mse_two_loss,
                    "structural_scale_coefficient": config["loss"][
                        "structural_scale"
                    ],
                }
                if raw_direction_mse
                else {}
            ),
            **(
                {
                    "direction_contrastive_kind": config["direction_contrastive"]["kind"],
                    "direction_contrastive_temperature": config["direction_contrastive"][
                        "temperature"
                    ],
                    "direction_contrastive_coefficient": config["direction_contrastive"][
                        "coefficient"
                    ],
                    "direction_contrastive_target_detached": config[
                        "direction_contrastive"
                    ]["target_detached"],
                    "direction_contrastive_group_reduction": config[
                        "direction_contrastive"
                    ]["group_reduction"],
                    "direction_contrastive_minimum_group_size": config[
                        "direction_contrastive"
                    ]["minimum_group_size"],
                }
                if direction_infonce
                else {}
            ),
            **(
                {
                    "direction_gauge_constraint_kind": config[
                        "direction_gauge_constraint"
                    ]["kind"],
                    "direction_gauge_frobenius_radius": config[
                        "direction_gauge_constraint"
                    ]["frobenius_radius"],
                    "direction_gauge_retraction_after_optimizer_step": config[
                        "direction_gauge_constraint"
                    ]["retraction_after_optimizer_step"],
                    "direction_gauge_project_first_moment_tangent": config[
                        "direction_gauge_constraint"
                    ]["project_first_moment_tangent"],
                    "direction_gauge_weight_decay": config[
                        "direction_gauge_constraint"
                    ]["weight_decay"],
                }
                if direction_gauge_fixed
                else {}
            ),
        }
    )
    key = experiment.get_key()
    url = f"https://www.comet.com/{workspace}/{args.project.replace('_', '-')}/{key}"
    info_path = run_root / "comet_experiment.json"
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"experiment_key": key, "url": url, "workspace": workspace, "project": args.project},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    tmp.replace(info_path)
    print(f"[comet-sidecar] experiment={url}", flush=True)

    stop = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    seen_metric_steps: set[int] = set()
    seen_gradient_steps: set[int] = set()
    seen_probe_steps: set[int] = set()
    try:
        while not stop:
            for row in _read_rows(metrics_path):
                step = int(row["step"])
                if step in seen_metric_steps:
                    continue
                metrics = {
                    f"train/{key}": value
                    for key, value in row.items()
                    if key not in {"schema", "step", "committed_logical_index"}
                    and isinstance(value, (int, float))
                }
                experiment.log_metrics(metrics, step=step)
                seen_metric_steps.add(step)
            for row in _read_rows(gradients_path):
                step = int(row["step"])
                if step in seen_gradient_steps:
                    continue
                metrics = _gradient_row_metrics(
                    row, raw_direction_mse=raw_direction_mse
                )
                experiment.log_metrics(metrics, step=step)
                seen_gradient_steps.add(step)
            for row in _read_rows(probe_path):
                step = int(row["step"])
                if step in seen_probe_steps:
                    continue
                metrics = {
                    f"probe/{key}": float(value)
                    for key, value in row.items()
                    if key not in {"schema", "step"}
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                experiment.log_metrics(metrics, step=step)
                seen_probe_steps.add(step)
            time.sleep(5.0)
    finally:
        experiment.end()


if __name__ == "__main__":
    main()
