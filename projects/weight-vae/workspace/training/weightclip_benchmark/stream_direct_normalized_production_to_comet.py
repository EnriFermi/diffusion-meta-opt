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


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.resolve()
    metrics_path = run_root / "train_metrics.jsonl"
    gradients_path = run_root / "gradient_telemetry.jsonl"
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
    categorical_gptq = schema == (
        "weightclip_parallel_categorical_gptq_700m_production_v1"
    )
    latent_rooted = schema == (
        "weightclip_direct_normalized_scaled_700m_"
        "latent_rooted_polar_tails_production_v1"
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
    is_polar = latent_rooted or latent_anticollapse or schema == (
        "weightclip_direct_normalized_scaled_700m_polar_tails_production_v1"
    )
    parameter_count = (
        696_787_328
        if categorical_gptq
        else (
            742_119_297
            if direction_gauge_fixed
            else (742_120_833 if is_polar else 706_301_312)
        )
    )
    architecture = (
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
    tags = ["production", "500k", "direct-normalized", "p32", "700m"]
    if is_polar:
        tags.extend(["polar-tails", "coordinate-owned-gradient-routing"])
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
            "steps": config["steps"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "scheduler": config["scheduler"],
            "weight_decay": config["weight_decay"],
            "grad_clip_norm": config["grad_clip_norm"],
            "encoder_depth": model["encoder_depth"],
            "decoder_depth": model["decoder_depth"],
            "hidden_dim": model["hidden_dim"],
            "latent_shape": f"{model['latent_slots']}x{model['latent_dim']}",
            "values_per_token": model["values_per_token"],
            "run_root": str(run_root),
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
                metrics: dict[str, float] = {}
                for name, value in row.items():
                    if name in {
                        "schema",
                        "step",
                        "groups",
                        "direction_scale_component_gradients",
                        "latent_objective_gradients",
                        "latent_code_scale_gradients",
                    }:
                        continue
                    if isinstance(value, (int, float)):
                        metrics[f"gradient/{name}"] = float(value)
                for group, payload in row["groups"].items():
                    if isinstance(payload, dict) and "gradient_rms" in payload:
                        metrics[f"grad_rms/{group}"] = float(payload["gradient_rms"])
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
                            name.startswith("direction_contrastive")
                            and isinstance(value, (int, float))
                            and not isinstance(value, bool)
                        ):
                            metrics[
                                f"direction_contrastive_gradient/{parameter}/{name}"
                            ] = float(value)
                experiment.log_metrics(metrics, step=step)
                seen_gradient_steps.add(step)
            time.sleep(5.0)
    finally:
        experiment.end()


if __name__ == "__main__":
    main()
