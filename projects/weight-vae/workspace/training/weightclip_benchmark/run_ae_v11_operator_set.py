from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.run_ae_v10_operator_set import (
    _build_worker_config as _build_v10_worker_config,
)


SCHEMA = "weightclip_ae_v11_exact64_four_trunk_complement_v1"
EXPECTED_TOTAL_PARAMETERS = 717_890_689
EXPECTED_TRAINABLE_PARAMETERS = 235_073_152
EXPECTED_ACTIVE_ENCODER_PARAMETERS = 68_488_576


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fresh exact64 V11 four-trunk complement training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "conf/weightclip_benchmark/"
            "ae_v11_exact64_four_trunk_complement_1984step.yaml"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "base_run_config",
        "output_root",
        "run_id",
        "seed",
        "layer_key",
        "operator_count",
        "steps",
        "physical_batch",
        "grad_accum_steps",
        "eval_every_steps",
        "eval_batch_size",
        "log_every",
        "comet_enabled",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("V11 exact64 config keys do not match the frozen schema")
    if payload["schema"] != SCHEMA:
        raise ValueError(f"V11 exact64 schema must be {SCHEMA!r}")
    frozen = {
        "operator_count": 64,
        "steps": 1984,
        "physical_batch": 6,
        "grad_accum_steps": 3,
        "eval_every_steps": 256,
        "eval_batch_size": 8,
        "seed": 42,
    }
    for key, value in frozen.items():
        if int(payload[key]) != value:
            raise ValueError(f"V11 exact64 requires {key}={value}")
    if str(payload["layer_key"]) != "layer3.0.conv2.weight":
        raise ValueError("V11 exact64 fixes layer3.0.conv2.weight")
    return payload


def _build_worker_config(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    cfg, output_root, manifest = _build_v10_worker_config(spec)
    cfg["model"]["big_vae"].update(
        {
            "architecture_version": "four_trunk_complement_v11",
            "v11_num_trunks": 4,
            "v11_blocks_per_trunk": 10,
            "v11_trunk_dim": 16,
        }
    )
    cfg["train"]["v11_complement_loss_weight"] = 0.1
    cfg["train"]["telemetry"]["comet"]["tags"] = [
        "v11",
        "exact64",
        "four-trunk",
        "complement-aux0p1",
    ]
    overfit = cfg["train"]["operator_bank"]["operator_set_overfit"]
    overfit.update(
        {
            "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
            "expected_active_encoder_parameters": EXPECTED_ACTIVE_ENCODER_PARAMETERS,
        }
    )
    cfg["train"]["telemetry"]["grad_layer_monitor"]["include_prefixes"] = [
        "distribution_encoder",
        "four_trunk_complement_encoder_v11",
        "mandatory_latent_bridge",
        "decoder_layers",
        "v11_residual_head",
    ]
    return cfg, output_root, manifest


def main() -> None:
    args = _parse_args()
    spec = _load_spec(args.config.expanduser().resolve())
    cfg, output_root, manifest = _build_worker_config(spec)
    summary = {
        "schema": SCHEMA,
        "architecture": "four_trunk_complement_v11",
        "device": cfg["train"]["device"],
        "dtype": cfg["train"].get("amp_dtype", "bf16"),
        "seed": cfg["data"]["seed"],
        "output_root": str(output_root),
        "steps": 1984,
        "physical_batch": 6,
        "grad_accum_steps": 3,
        "latent_contract": "[B,32,384]=zc[320]+concat(4 trunks x 16)",
        "checkpoint_policy": "one final step_0001984.pt; no latest; no resume-state",
        "parameter_contract": {
            "total": EXPECTED_TOTAL_PARAMETERS,
            "trainable": EXPECTED_TRAINABLE_PARAMETERS,
            "active_encoder": EXPECTED_ACTIVE_ENCODER_PARAMETERS,
            "routing_blocks": 40,
            "trunks": 4,
            "blocks_per_trunk": 10,
        },
        "selection": manifest,
    }
    print("[v11-exact64] stage=preflight", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[v11-exact64] stage=dry-run-complete no_files_written=true", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=False)
    Path(cfg["training_artifacts"]["run_root_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    resolved_path = output_root / "resolved_run_config.json"
    selection_path = output_root / "operator_set_selection.json"
    resolved_path.write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        f"[v11-exact64] stage=train resolved={resolved_path} selection={selection_path}",
        flush=True,
    )
    _run_worker(
        rank=0,
        world_size=1,
        cfg_dict=cfg,
        master_addr="127.0.0.1",
        master_port=0,
        monitor_queue=None,
    )
    checkpoint_root = Path(cfg["train"]["checkpoint_dir"]) / "stage_1"
    expected = checkpoint_root / "step_0001984.pt"
    if not expected.is_file() or (checkpoint_root / "latest.pt").exists():
        raise RuntimeError("V11 exact64 final-only checkpoint contract failed")
    metrics_path = Path(
        cfg["train"]["operator_bank"]["operator_set_overfit"]["metrics_path"]
    )
    final_metrics = json.loads(
        metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    )
    print(
        "[v11-exact64] stage=complete "
        f"checkpoint={expected} "
        f"scientific_contract_pass={bool(final_metrics['v11_final_scientific_contract_pass'])} "
        f"failures={final_metrics['v11_final_scientific_contract_failures']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
