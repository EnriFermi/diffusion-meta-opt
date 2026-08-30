from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from big_vae.weightclip_benchmark.manifests import write_json_immutable
from dataset.logging_utils import LOG_PATH_ENV
from training.big_vae.epsilon_fork import (
    EPSILON_FORK_ADDITIONAL_STEPS,
    EPSILON_FORK_SCHEMA,
    EPSILON_FORK_SOURCE_EPSILON,
    EPSILON_FORK_SOURCE_STEP,
    EPSILON_FORK_TARGET_EPSILONS,
    stable_resume_checkpoint_identity,
)
from training.big_vae.worker import _run_worker
from training.weightclip_benchmark.run_ae_two_operator_overfit import (
    _build_worker_config as _build_overfit_worker_config,
)
from training.weightclip_benchmark.run_ae_two_operator_overfit import _load_overfit_spec


_SCHEMA = "weightclip_ae_two_operator_epsilon_fork_launcher_v1"
_ARM_NAMES = ("eps_1e-8", "eps_1e-12")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fork the exact two-operator step2500 full resume state for 250 optimizer steps, "
            "changing only the loaded AdamW parameter-group epsilon."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/weightclip_benchmark/ae_v5_two_operator_epsilon_fork.yaml"),
    )
    parser.add_argument("--arm", choices=_ARM_NAMES, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate the full contract without writing files.")
    return parser.parse_args()


def _load_fork_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"epsilon-fork config must use schema {_SCHEMA!r}")
    expected_keys = {
        "schema",
        "base_overfit_config",
        "source_resume_checkpoint",
        "source_resume_sha256",
        "source_step",
        "additional_steps",
        "checkpoint_every",
        "log_every",
        "eval_every_steps",
        "control_reference",
        "arms",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            f"epsilon-fork config keys must be exactly {sorted(expected_keys)}, got {sorted(payload)}"
        )
    if int(payload["source_step"]) != EPSILON_FORK_SOURCE_STEP:
        raise ValueError("epsilon-fork source_step must be exactly 2500")
    if int(payload["additional_steps"]) != EPSILON_FORK_ADDITIONAL_STEPS:
        raise ValueError("epsilon-fork additional_steps must be exactly 250")
    if int(payload["checkpoint_every"]) != 250:
        raise ValueError("epsilon-fork checkpoint_every must be exactly 250")
    if int(payload["log_every"]) != 10:
        raise ValueError("epsilon-fork log_every must remain exactly 10 for the step2510 replay gate")
    if int(payload["eval_every_steps"]) != 25:
        raise ValueError("epsilon-fork per-operator evaluation interval must be exactly 25 steps")
    source_sha = str(payload["source_resume_sha256"])
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("epsilon-fork source_resume_sha256 must be lowercase hexadecimal")

    control = payload["control_reference"]
    expected_control = {"step", "absolute_tolerance", "metrics"}
    if not isinstance(control, Mapping) or set(control) != expected_control:
        raise ValueError(f"control_reference keys must be exactly {sorted(expected_control)}")

    arms = payload["arms"]
    if not isinstance(arms, Mapping) or set(arms) != set(_ARM_NAMES):
        raise ValueError(f"epsilon-fork arms must be exactly {list(_ARM_NAMES)}")
    expected_arm_keys = {"target_optimizer_epsilon", "output_root", "run_id", "comet_enabled"}
    observed_eps: list[float] = []
    roots: list[Path] = []
    for arm_name in _ARM_NAMES:
        arm = arms[arm_name]
        if not isinstance(arm, Mapping) or set(arm) != expected_arm_keys:
            raise ValueError(f"arm {arm_name!r} keys must be exactly {sorted(expected_arm_keys)}")
        target_eps = float(arm["target_optimizer_epsilon"])
        if target_eps not in EPSILON_FORK_TARGET_EPSILONS:
            raise ValueError(f"arm {arm_name!r} has invalid target epsilon {target_eps}")
        observed_eps.append(target_eps)
        roots.append(Path(str(arm["output_root"])).expanduser().resolve())
        if not str(arm["run_id"]).strip():
            raise ValueError(f"arm {arm_name!r} run_id must be non-empty")
    if observed_eps != list(EPSILON_FORK_TARGET_EPSILONS):
        raise ValueError(f"epsilon-fork arms must map in order to {EPSILON_FORK_TARGET_EPSILONS}")
    if len(set(roots)) != 2:
        raise ValueError("epsilon-fork arms require distinct output roots")
    return payload


def _build_fork_worker_config(
    spec: Mapping[str, Any],
    *,
    arm_name: str,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    if arm_name not in _ARM_NAMES:
        raise ValueError(f"unknown epsilon-fork arm: {arm_name!r}")
    source_path = Path(str(spec["source_resume_checkpoint"])).expanduser().resolve(strict=True)
    source_identity = stable_resume_checkpoint_identity(source_path)
    if source_identity["sha256"] != str(spec["source_resume_sha256"]):
        raise RuntimeError(
            "epsilon-fork source resume SHA256 mismatch: "
            f"expected={spec['source_resume_sha256']} actual={source_identity['sha256']}"
        )

    base_path = Path(str(spec["base_overfit_config"])).expanduser().resolve(strict=True)
    base_spec = copy.deepcopy(_load_overfit_spec(base_path))
    arm = dict(spec["arms"][arm_name])
    base_spec["output_root"] = str(Path(str(arm["output_root"])).expanduser().resolve())
    base_spec["run_id"] = str(arm["run_id"])
    base_spec["checkpoint_every"] = int(spec["checkpoint_every"])
    base_spec["log_every"] = int(spec["log_every"])
    base_spec["eval_every_steps"] = int(spec["eval_every_steps"])
    base_spec["comet_enabled"] = bool(arm["comet_enabled"])
    base_spec["early_success"] = dict(base_spec["early_success"])
    base_spec["early_success"]["enabled"] = False
    cfg, output_root, operator_manifest = _build_overfit_worker_config(base_spec)

    train = cfg["train"]
    stage_dir = Path(str(train["checkpoint_dir"])) / "stage_1"
    stop_after_step = EPSILON_FORK_SOURCE_STEP + EPSILON_FORK_ADDITIONAL_STEPS
    train.update(
        {
            "stop_after_step": stop_after_step,
            "checkpoint_every": int(spec["checkpoint_every"]),
        }
    )
    resume = train["resume_state"]
    resume.update(
        {
            "enabled": True,
            "auto_resume": False,
            "explicit_checkpoint": str(source_path),
            "load_model_state": True,
            "load_optimizer_state": True,
            "load_scheduler_state": True,
            "load_scaler_state": True,
            "load_rng_state": True,
            "load_step": True,
            "save_every": int(spec["checkpoint_every"]),
            "dir": str(stage_dir / "resume_state"),
        }
    )
    overfit = train["operator_bank"]["two_operator_overfit"]
    overfit["eval_every_steps"] = int(spec["eval_every_steps"])
    overfit["early_success"]["enabled"] = False
    target_eps = float(arm["target_optimizer_epsilon"])
    train["exact_optimizer_epsilon_fork"] = {
        "schema": EPSILON_FORK_SCHEMA,
        "enabled": True,
        "source_resume_checkpoint": str(source_path),
        "source_resume_sha256": str(source_identity["sha256"]),
        "source_resume_stat": dict(source_identity["stat"]),
        "expected_source_step": EPSILON_FORK_SOURCE_STEP,
        "additional_optimizer_steps": EPSILON_FORK_ADDITIONAL_STEPS,
        "source_optimizer_epsilon": EPSILON_FORK_SOURCE_EPSILON,
        "target_optimizer_epsilon": target_eps,
        "expected_optimizer_group_count": 1,
        "expected_optimizer_parameter_counts": [968],
        "expected_optimizer_state_count": 960,
        "startup_ledger_path": str(stage_dir / "epsilon_fork_startup_ledger.json"),
        "control_replay_path": str(stage_dir / "epsilon_fork_control_replay_step2510.json"),
        "control_reference": copy.deepcopy(dict(spec["control_reference"])),
        "suppress_resume_writes": True,
    }
    train["telemetry"]["comet"]["tags"] = [
        "two-full-operators",
        "v5",
        "exact-resume",
        "epsilon-fork",
        arm_name,
    ]

    if float(train["eps"]) != EPSILON_FORK_SOURCE_EPSILON:
        raise AssertionError("both epsilon-fork configs must build the source optimizer at eps=1e-8")
    if int(train["slice_batch_size"]) != 18 or int(train["grad_accum_steps"]) != 1:
        raise AssertionError("epsilon-fork worker config drifted from B18 x accum1")
    if int(train["max_steps"]) != 500_000 or int(train["stop_after_step"]) != 2_750:
        raise AssertionError("epsilon-fork must retain the 500k scheduler horizon and stop at step2750")
    if any(not bool(resume[key]) for key in (
        "load_model_state",
        "load_optimizer_state",
        "load_scheduler_state",
        "load_scaler_state",
        "load_rng_state",
        "load_step",
    )):
        raise AssertionError("epsilon-fork must restore every training-state component")
    if output_root == source_path.parent or source_path.is_relative_to(output_root):
        raise RuntimeError("epsilon-fork output root must not contain or overlap its immutable source resume")
    return cfg, output_root, operator_manifest, source_identity


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve(strict=True)
    spec = _load_fork_spec(config_path)
    cfg, output_root, operator_manifest, source_identity = _build_fork_worker_config(
        spec,
        arm_name=args.arm,
    )
    fork = cfg["train"]["exact_optimizer_epsilon_fork"]
    summary = {
        "schema": _SCHEMA,
        "arm": args.arm,
        "source": source_identity,
        "source_step": EPSILON_FORK_SOURCE_STEP,
        "stop_after_step": cfg["train"]["stop_after_step"],
        "additional_optimizer_steps": EPSILON_FORK_ADDITIONAL_STEPS,
        "source_optimizer_epsilon": EPSILON_FORK_SOURCE_EPSILON,
        "target_optimizer_epsilon": fork["target_optimizer_epsilon"],
        "physical_batch": cfg["train"]["slice_batch_size"],
        "grad_accum_steps": cfg["train"]["grad_accum_steps"],
        "committed_logical_range": [45_000, 49_500],
        "per_operator_eval_every_steps": cfg["train"]["operator_bank"]["two_operator_overfit"][
            "eval_every_steps"
        ],
        "rolling_resume_writes_suppressed": True,
        "output_root": str(output_root),
        "operator_manifest": operator_manifest,
    }
    print("[epsilon-fork] stage=preflight", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[epsilon-fork] stage=dry-run-complete no_files_written=true", flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=False)
    run_root = Path(cfg["training_artifacts"]["run_root_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_run_config.json"
    manifest_path = output_root / "two_full_operator_manifest.json"
    launch_contract_path = output_root / "epsilon_fork_launch_contract.json"
    write_json_immutable(resolved_path, cfg)
    write_json_immutable(manifest_path, operator_manifest)
    write_json_immutable(launch_contract_path, summary)
    os.environ[LOG_PATH_ENV] = str(cfg["logging"]["file_path"])
    print(
        f"[epsilon-fork] stage=train arm={args.arm} resolved={resolved_path} contract={launch_contract_path}",
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
    checkpoint = Path(cfg["train"]["checkpoint_dir"]) / "stage_1" / "step_0002750.pt"
    ledger = Path(str(fork["startup_ledger_path"]))
    if not checkpoint.is_file() or not ledger.is_file():
        raise RuntimeError(f"epsilon-fork arm lacks final evidence: checkpoint={checkpoint} ledger={ledger}")
    if float(fork["target_optimizer_epsilon"]) == EPSILON_FORK_SOURCE_EPSILON:
        replay = Path(str(fork["control_replay_path"]))
        if not replay.is_file():
            raise RuntimeError(f"epsilon=1e-8 arm lacks the mandatory step2510 replay artifact: {replay}")
    print(
        f"[epsilon-fork] stage=complete arm={args.arm} checkpoint={checkpoint} ledger={ledger}",
        flush=True,
    )


if __name__ == "__main__":
    main()
