from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _patch_argparse_lazy_help_for_hydra_py314() -> None:
    if getattr(argparse.ArgumentParser, "_hydra_lazy_help_py314_patch", False):
        return
    original_check_help = getattr(argparse.ArgumentParser, "_check_help", None)
    if original_check_help is None:
        return

    def patched_check_help(self: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        original_check_help(self, action)

    argparse.ArgumentParser._check_help = patched_check_help  # type: ignore[method-assign]
    argparse.ArgumentParser._hydra_lazy_help_py314_patch = True  # type: ignore[attr-defined]


_patch_argparse_lazy_help_for_hydra_py314()


def _plain(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved config must be a mapping")
    return payload


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip().lower()).strip("-")
    return text or "run"


def bool_str(value: Any) -> str:
    return "true" if bool(value) else "false"


def resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def command_to_text(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def env_to_text(env: Mapping[str, str] | None) -> str:
    if not env:
        return ""
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))


def command_with_env_to_text(command: Sequence[str], env: Mapping[str, str] | None) -> str:
    env_text = env_to_text(env)
    command_text = command_to_text(command)
    return f"{env_text} {command_text}" if env_text else command_text


def checkpoint_label(path: Path, explicit: str = "") -> str:
    if explicit.strip():
        return slugify(explicit)
    parts = [path.parent.parent.name, path.parent.name, path.stem]
    return slugify("_".join(part for part in parts if part))


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("big_vae_eval_suite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    fh = logging.FileHandler(run_dir / "suite.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


@dataclass(slots=True)
class StageResult:
    name: str
    enabled: bool
    ok: bool
    returncode: int | None = None
    duration_s: float = 0.0
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    output_dir: str = ""
    log_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ok": self.ok,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "command": self.command,
            "env": self.env,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "error": self.error,
        }


class EvalSuite:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        suite_cfg = _mapping(cfg.get("suite", {}), "suite")
        checkpoint_cfg = _mapping(cfg.get("checkpoint", {}), "checkpoint")
        self.fail_fast = bool(suite_cfg.get("fail_fast", True))
        self.dry_run = bool(suite_cfg.get("dry_run", False))
        suite_python = str(suite_cfg.get("python", "")).strip()
        self.python = suite_python or sys.executable
        self.python_explicit = bool(suite_python)
        self.conda_env = str(suite_cfg.get("conda_env", "")).strip()
        self.conda_executable = str(suite_cfg.get("conda_executable", "conda")).strip() or "conda"
        checkpoint_raw = str(checkpoint_cfg.get("big_vae", "")).strip()
        if not checkpoint_raw:
            raise ValueError("checkpoint.big_vae must be set")
        self.checkpoint_path = resolve_path(checkpoint_raw)
        if not self.dry_run and not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint.big_vae file does not exist: {self.checkpoint_path}")
        self.prior_checkpoint = self.optional_existing_file(
            checkpoint_cfg.get("latent_diffusion_prior", ""),
            "checkpoint.latent_diffusion_prior",
        )
        self.label = checkpoint_label(self.checkpoint_path, str(checkpoint_cfg.get("label", "")))
        default_root = os.environ.get("BIG_VAE_EVAL_SUITE_ROOT") or os.environ.get(
            "BIG_VAE_ARTIFACT_ROOT",
            "./artifacts/big_vae",
        ).rstrip("/") + "/eval/suite"
        self.root_dir = resolve_path(str(suite_cfg.get("root_dir", default_root)))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_label = slugify(str(suite_cfg.get("run_label", "")).strip() or self.label)
        base_run_id = f"{stamp}__{run_label}"
        self.run_id, self.run_dir = self.reserve_run_dir(base_run_id)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(self.run_dir)
        self.summary_path = self.run_dir / "summary.json"
        self.results: list[StageResult] = []

    def reserve_run_dir(self, base_run_id: str) -> tuple[str, Path]:
        runs_root = self.root_dir / "runs"
        for suffix in range(100):
            run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
            run_dir = runs_root / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_id, run_dir
            except FileExistsError:
                continue
        raise FileExistsError(f"Could not reserve unique suite run directory under {runs_root}: {base_run_id}")

    def optional_existing_file(self, value: Any, label: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = resolve_path(text)
        if not self.dry_run and not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        return str(path)

    def python_command(self, stage_cfg: Mapping[str, Any] | None = None) -> list[str]:
        stage_cfg = stage_cfg or {}
        stage_python = str(stage_cfg.get("python", "")).strip()
        if stage_python:
            return [stage_python]
        stage_conda_env = str(stage_cfg.get("conda_env", "")).strip()
        if stage_conda_env:
            if not self.dry_run and shutil.which(self.conda_executable) is None:
                raise FileNotFoundError(f"conda executable not found: {self.conda_executable}")
            return [self.conda_executable, "run", "--no-capture-output", "-n", stage_conda_env, "python"]
        if self.conda_env and not self.python_explicit:
            if not self.dry_run and shutil.which(self.conda_executable) is None:
                raise FileNotFoundError(f"conda executable not found: {self.conda_executable}")
            return [self.conda_executable, "run", "--no-capture-output", "-n", self.conda_env, "python"]
        return [self.python]

    def write_snapshots(self) -> None:
        (self.run_dir / "config_resolved.json").write_text(
            json.dumps(self.cfg, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.run_dir / "suite_context.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "run_dir": str(self.run_dir),
                    "checkpoint": str(self.checkpoint_path),
                    "checkpoint_label": self.label,
                    "latent_diffusion_prior": self.prior_checkpoint,
                    "python": self.python,
                    "conda_env": self.conda_env,
                    "conda_executable": self.conda_executable,
                    "dry_run": self.dry_run,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def update_summary(self) -> None:
        payload = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_label": self.label,
            "latent_diffusion_prior": self.prior_checkpoint,
            "ok": all(result.ok for result in self.results if result.enabled),
            "stages": [result.to_dict() for result in self.results],
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def run_command(
        self,
        *,
        name: str,
        command: list[str],
        output_dir: Path,
        env: Mapping[str, str] | None = None,
    ) -> StageResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{slugify(name)}.log"
        result = StageResult(
            name=name,
            enabled=True,
            ok=True,
            command=command,
            env={str(key): str(value) for key, value in (env or {}).items()},
            output_dir=str(output_dir),
            log_path=str(log_path),
        )
        self.logger.info("Stage %s starting: output=%s", name, output_dir)
        command_text = command_with_env_to_text(command, result.env)
        self.logger.info("Command: %s", command_text)
        start = time.time()
        if self.dry_run:
            result.duration_s = 0.0
            log_path.write_text("DRY RUN\n" + command_text + "\n", encoding="utf-8")
            self.results.append(result)
            self.update_summary()
            return result
        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(key): str(value) for key, value in env.items()})
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write("$ " + command_text + "\n\n")
            fh.flush()
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=merged_env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result.duration_s = time.time() - start
        result.returncode = int(completed.returncode)
        result.ok = completed.returncode == 0
        if not result.ok:
            result.error = f"stage exited with code {completed.returncode}; see {log_path}"
            self.logger.error("Stage %s failed after %.1fs: %s", name, result.duration_s, result.error)
        else:
            self.logger.info("Stage %s finished in %.1fs", name, result.duration_s)
        self.results.append(result)
        self.update_summary()
        if not result.ok and self.fail_fast:
            raise RuntimeError(result.error)
        return result

    def stage_disabled(self, name: str) -> None:
        self.results.append(StageResult(name=name, enabled=False, ok=True))
        self.update_summary()

    def run_heldout(self) -> None:
        heldout = _mapping(_mapping(self.cfg.get("stages", {}), "stages").get("heldout_eval", {}), "stages.heldout_eval")
        if not bool(heldout.get("enabled", True)):
            self.stage_disabled("heldout_eval")
            return
        output_dir = self.run_dir / "heldout_eval"
        heldout_root = resolve_path(str(heldout.get("heldout_root", "")))
        if not self.dry_run and not (heldout_root / "manifest.json").exists():
            raise FileNotFoundError(
                f"stages.heldout_eval.heldout_root manifest not found: {heldout_root / 'manifest.json'}"
            )
        env = {
            "BIG_VAE_CHECKPOINT": str(self.checkpoint_path),
            "HELDOUT_ROOT": str(heldout_root),
            "HELDOUT_LOG_DIR": str(self.logs_dir / "heldout_runtime"),
            "EVAL_OUTPUT_DIR": str(output_dir),
            "EVAL_DEVICE": str(heldout.get("device", "auto")),
            "EVAL_BATCH_SIZE": str(heldout.get("batch_size", 256)),
            "EVAL_MAX_RECORDS": str(heldout.get("max_records", 0)),
            "EVAL_MAX_SLICES_PER_SOURCE": str(heldout.get("max_slices_per_source", 0)),
            "EVAL_SAVE_RECORD_METRICS": bool_str(heldout.get("save_record_metrics", True)),
            "EVAL_LOG_EVERY_RECORDS": str(heldout.get("log_every_records", 100)),
            "EVAL_REQUIRE_FULL_COVERAGE": bool_str(heldout.get("require_full_coverage", True)),
            "EVAL_LATENT_DUMP_ENABLED": bool_str(heldout.get("latent_dump_enabled", True)),
            "EVAL_LATENT_DUMP_MAX_ENTRIES": str(heldout.get("latent_dump_max_entries", 256)),
            "EVAL_LATENT_DUMP_MAX_SLICES_PER_SOURCE": str(heldout.get("latent_dump_max_slices_per_source", 1)),
            "EVAL_LATENT_DUMP_BALANCE_ENABLED": bool_str(heldout.get("latent_dump_balance_enabled", True)),
            "EVAL_LATENT_DUMP_BALANCE_KEYS": str(
                heldout.get("latent_dump_balance_keys", "dataset,model,layer_type,depth_label")
            ),
            "EVAL_LATENT_DUMP_MAX_PER_GROUP": str(heldout.get("latent_dump_max_per_group", 1)),
            "EVAL_LATENT_PLOT_ENABLED": bool_str(heldout.get("latent_plot_enabled", True)),
            "EVAL_LATENT_PLOT_TSNE": bool_str(heldout.get("latent_plot_tsne", True)),
        }
        self.run_command(
            name="heldout_eval",
            command=[
                *self.python_command(heldout),
                "post_train_research/big_vae_heldout_eval/evaluate_big_vae_heldout.py",
            ],
            output_dir=output_dir,
            env=env,
        )

    def scaling_overrides(
        self,
        *,
        common: Mapping[str, Any],
        job: Mapping[str, Any],
        setup_kind: str,
        output_root: Path,
    ) -> list[str]:
        init_common = _mapping(common.get("init", {}), "scaling.common.init")
        train_common = _mapping(common.get("train", {}), "scaling.common.train")
        data_common = _mapping(common.get("data", {}), "scaling.common.data")
        logging_common = _mapping(common.get("logging", {}), "scaling.common.logging")
        setup_common = _mapping(common.get("setup", {}), "scaling.common.setup")

        init_kind = str(job.get("init_kind", init_common.get("kind", "diffusion_prior"))).strip().lower()
        if setup_kind == "raw":
            init_kind = str(job.get("raw_init_kind", "fresh")).strip().lower()
        raw_prior = str(job.get("diffusion_prior_checkpoint", init_common.get("diffusion_prior_checkpoint", "")) or "").strip()
        prior = (
            self.optional_existing_file(raw_prior, "stages.scaling_check diffusion prior checkpoint")
            if raw_prior
            else self.prior_checkpoint
        )
        if setup_kind == "latent" and init_kind == "diffusion_prior" and not prior:
            raise ValueError(f"scaling job {job.get('label', job.get('profile'))}: diffusion prior checkpoint is required")

        label_bits = [
            self.label,
            str(job.get("label", job.get("profile", "profile"))),
            setup_kind,
            init_kind,
        ]
        run_label = slugify("_".join(label_bits))
        shared_label = run_label
        profile = str(job.get("profile", "cifar10_50k")).strip()
        max_steps = int(job.get("max_steps", train_common.get("max_steps", 0)))
        eval_every = int(job.get("eval_every_steps", logging_common.get("eval_every_steps", 0)))
        batch_size = int(job.get("batch_size", data_common.get("batch_size", 0)))
        eval_batch_size = int(job.get("eval_batch_size", data_common.get("eval_batch_size", 0)))
        lr = float(job.get("lr", train_common.get("lr", 1e-3)))
        optimizer = str(job.get("optimizer_name", train_common.get("optimizer_name", "AdamW")))

        overrides = [
            f"experiment.run_label={run_label}",
            f"experiment.profile={profile}",
            f"experiment.notes={str(common.get('notes', ''))}",
            f"storage.root_dir={output_root}",
            f"storage.shared_checkpoint_root_dir={output_root / 'shared_checkpoints'}",
            f"storage.shared_checkpoint_label={shared_label}",
            f"storage.checkpoint_every_steps={int(common.get('checkpoint_every_steps', 250))}",
            f"setup.kind={setup_kind}",
            f"setup.big_vae_checkpoint={self.checkpoint_path if setup_kind == 'latent' else ''}",
            f"setup.big_vae_decode={str(setup_common.get('big_vae_decode', 'weights'))}",
            f"setup.big_vae_tile_T_patches={int(setup_common.get('big_vae_tile_T_patches', 4))}",
            f"setup.big_vae_tile_d_out={int(setup_common.get('big_vae_tile_d_out', 64))}",
            f"setup.big_vae_latent_parameterization={str(setup_common.get('big_vae_latent_parameterization', 'euclidean'))}",
            f"setup.big_vae_latent_noise_std={float(setup_common.get('big_vae_latent_noise_std', 0.0))}",
            f"init.kind={init_kind}",
            f"init.fresh_latent_mode={str(init_common.get('fresh_latent_mode', 'random'))}",
            f"init.random_init_std={float(init_common.get('random_init_std', 0.02))}",
            f"init.source_run_dir={str(init_common.get('source_run_dir', ''))}",
            f"init.source_checkpoint={str(init_common.get('source_checkpoint', 'best'))}",
            f"init.source_prefer_direct_latent={bool_str(init_common.get('source_prefer_direct_latent', True))}",
            f"init.diffusion_prior_checkpoint={prior if setup_kind == 'latent' else ''}",
            f"init.diffusion_prior_steps={int(init_common.get('diffusion_prior_steps', 50))}",
            f"init.diffusion_prior_sampler={str(init_common.get('diffusion_prior_sampler', 'ddim'))}",
            f"init.diffusion_prior_eta={float(init_common.get('diffusion_prior_eta', 0.0))}",
            f"init.calibration_batches={int(init_common.get('calibration_batches', 1))}",
            f"data.data_dir={str(job.get('data_dir', data_common.get('data_dir', '')))}",
            f"data.train_subset={int(job.get('train_subset', data_common.get('train_subset', 0)))}",
            f"data.test_subset={int(job.get('test_subset', data_common.get('test_subset', 0)))}",
            f"data.batch_size={batch_size}",
            f"data.eval_batch_size={eval_batch_size}",
            f"data.num_workers={int(data_common.get('num_workers', 4))}",
            f"data.download={bool_str(data_common.get('download', True))}",
            f"train.device={str(train_common.get('device', 'auto'))}",
            f"train.seed={int(job.get('seed', train_common.get('seed', 42)))}",
            f"train.epochs={int(train_common.get('epochs', 1000))}",
            f"train.max_steps={max_steps}",
            f"train.optimizer_name={optimizer}",
            f"train.lr={lr}",
            f"train.weight_decay={float(job.get('weight_decay', train_common.get('weight_decay', 0.0)))}",
            f"train.adam_beta1={float(train_common.get('adam_beta1', 0.9))}",
            f"train.adam_beta2={float(train_common.get('adam_beta2', 0.999))}",
            f"train.adam_eps={float(train_common.get('adam_eps', 1e-9))}",
            f"train.grad_clip_norm={float(train_common.get('grad_clip_norm', 0.0))}",
            f"train.label_smoothing={float(train_common.get('label_smoothing', 0.0))}",
            f"train.amp={bool_str(train_common.get('amp', True))}",
            f"train.tf32={bool_str(train_common.get('tf32', True))}",
            f"train.compile={bool_str(train_common.get('compile', False))}",
            f"train.latent_lr_scheduler={str(train_common.get('latent_lr_scheduler', 'constant'))}",
            f"train.latent_lr_floor_ratio={float(train_common.get('latent_lr_floor_ratio', 0.1))}",
            f"train.latent_lr_decay_steps={int(train_common.get('latent_lr_decay_steps', 0))}",
            f"logging.log_every_steps={int(logging_common.get('log_every_steps', 50))}",
            f"logging.eval_every_steps={eval_every}",
            f"logging.latent_debug_metrics={bool_str(logging_common.get('latent_debug_metrics', setup_kind == 'latent'))}",
            f"logging.latent_jacobian_eps={float(logging_common.get('latent_jacobian_eps', 1e-3))}",
            f"logging.latent_jacobian_probes={int(logging_common.get('latent_jacobian_probes', 16))}",
            "telemetry.comet.enabled=false",
        ]
        return overrides

    def run_scaling(self) -> None:
        scaling = _mapping(_mapping(self.cfg.get("stages", {}), "stages").get("scaling_check", {}), "stages.scaling_check")
        if not bool(scaling.get("enabled", True)):
            self.stage_disabled("scaling_check")
            return
        jobs = _list(scaling.get("jobs", []), "stages.scaling_check.jobs")
        if not jobs:
            raise ValueError("stages.scaling_check.jobs must contain at least one profile job when enabled")
        output_root = self.run_dir / "scaling_check"
        common = _mapping(scaling.get("common", {}), "stages.scaling_check.common")
        run_latent = bool(scaling.get("run_latent", True))
        run_raw = bool(scaling.get("run_raw_baseline", False))
        for raw_job in jobs:
            job = _mapping(raw_job, "stages.scaling_check.jobs[]")
            setups: list[str] = []
            if run_latent and bool(job.get("run_latent", True)):
                setups.append("latent")
            if run_raw and bool(job.get("run_raw_baseline", True)):
                setups.append("raw")
            for setup_kind in setups:
                job_label = slugify(str(job.get("label", job.get("profile", "profile"))))
                output_dir = output_root / job_label / setup_kind
                overrides = self.scaling_overrides(common=common, job=job, setup_kind=setup_kind, output_root=output_dir)
                self.run_command(
                    name=f"scaling_{job_label}_{setup_kind}",
                    command=[
                        *self.python_command(scaling),
                        "post_train_research/vit_latent_scaling/main.py",
                        *overrides,
                    ],
                    output_dir=output_dir,
                )

    def run_landscape_ablation(self) -> None:
        ablation = _mapping(
            _mapping(self.cfg.get("stages", {}), "stages").get("landscape_ablation", {}),
            "stages.landscape_ablation",
        )
        if not bool(ablation.get("enabled", False)):
            self.stage_disabled("landscape_ablation")
            return
        latent_init = str(ablation.get("latent_init", "diffusion_prior")).strip().lower()
        raw_prior = str(ablation.get("diffusion_prior_checkpoint", "") or "").strip()
        prior_checkpoint = self.optional_existing_file(
            raw_prior,
            "stages.landscape_ablation.diffusion_prior_checkpoint",
        ) if raw_prior else self.prior_checkpoint
        if latent_init == "diffusion_prior" and not prior_checkpoint:
            raise ValueError("checkpoint.latent_diffusion_prior is required for landscape_ablation latent_init=diffusion_prior")
        output_dir = self.run_dir / "landscape_ablation"
        model = _mapping(ablation.get("model", {}), "stages.landscape_ablation.model")
        search = _mapping(ablation.get("search", {}), "stages.landscape_ablation.search")
        seeds = _list(ablation.get("seeds", [0]), "stages.landscape_ablation.seeds")
        checkpoint_steps = _list(
            ablation.get("checkpoint_steps", [1, 50, 100, 200, 500]),
            "checkpoint_steps",
        )
        radial_scales = _list(
            search.get("radial_scales", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
            "radial_scales",
        )
        theta_grid = _list(
            search.get("theta_grid", [-1.57079632679, -0.78539816339, 0.0, 0.78539816339, 1.57079632679]),
            "theta_grid",
        )
        for key, values in {
            "stages.landscape_ablation.seeds": seeds,
            "stages.landscape_ablation.checkpoint_steps": checkpoint_steps,
            "stages.landscape_ablation.search.radial_scales": radial_scales,
            "stages.landscape_ablation.search.theta_grid": theta_grid,
        }.items():
            if not values:
                raise ValueError(f"{key} must contain at least one value")
        command = [
            *self.python_command(ablation),
            "experiments/diagnose_latent_landscape_tinyvit.py",
            "--data_dir",
            str(ablation.get("data_dir", "./data/cifar10")),
            "--output_dir",
            str(output_dir),
            "--device",
            str(ablation.get("device", "auto")),
            "--seeds",
            *[str(seed) for seed in seeds],
            "--train_subset",
            str(ablation.get("train_subset", 10000)),
            "--test_subset",
            str(ablation.get("test_subset", 2000)),
            "--batch_size",
            str(ablation.get("batch_size", 128)),
            "--eval_batch_size",
            str(ablation.get("eval_batch_size", 256)),
            "--num_workers",
            str(ablation.get("num_workers", 4)),
            "--diag_batch_size",
            str(ablation.get("diag_batch_size", 1024)),
            "--diagnostic_split",
            str(ablation.get("diagnostic_split", "train")),
            "--steps",
            str(ablation.get("steps", 3000)),
            "--epochs",
            str(ablation.get("epochs", 1000)),
            "--lr",
            str(ablation.get("lr", 1e-2)),
            "--weight_decay",
            str(ablation.get("weight_decay", 0.0)),
            "--grad_clip_norm",
            str(ablation.get("grad_clip_norm", 0.0)),
            "--log_every",
            str(ablation.get("log_every", 25)),
            "--eval_every",
            str(ablation.get("eval_every", 100)),
            "--checkpoint_steps",
            *[str(step) for step in checkpoint_steps],
            "--vae_ckpt",
            str(self.checkpoint_path),
            "--big_vae_latent_init",
            latent_init,
            "--big_vae_latent_parameterization",
            str(ablation.get("latent_parameterization", "euclidean")),
            "--prior_ckpt",
            prior_checkpoint,
            "--big_vae_diffusion_prior_steps",
            str(ablation.get("diffusion_prior_steps", 50)),
            "--big_vae_diffusion_prior_sampler",
            str(ablation.get("diffusion_prior_sampler", "ddim")),
            "--big_vae_diffusion_prior_eta",
            str(ablation.get("diffusion_prior_eta", 0.0)),
            "--big_vae_decode",
            str(ablation.get("big_vae_decode", "weights")),
            "--big_vae_tile_T_patches",
            str(ablation.get("big_vae_tile_T_patches", 4)),
            "--big_vae_tile_d_out",
            str(ablation.get("big_vae_tile_d_out", 64)),
            "--big_vae_encoder_context_rows",
            str(ablation.get("big_vae_encoder_context_rows", 64)),
            "--big_vae_encoder_context_std",
            str(ablation.get("big_vae_encoder_context_std", 1.0)),
            "--big_vae_encoder_batch_size",
            str(ablation.get("big_vae_encoder_batch_size", 16)),
            "--big_vae_init_calibration_batches",
            str(ablation.get("calibration_batches", 1)),
            "--image_size",
            str(model.get("image_size", 32)),
            "--patch_size",
            str(model.get("patch_size", 8)),
            "--hidden_dim",
            str(model.get("hidden_dim", 64)),
            "--depth",
            str(model.get("depth", 1)),
            "--num_heads",
            str(model.get("num_heads", 4)),
            "--mlp_ratio",
            str(model.get("mlp_ratio", 2.0)),
            "--dropout",
            str(model.get("dropout", 0.0)),
            "--attention_dropout",
            str(model.get("attention_dropout", 0.0)),
            "--num_classes",
            str(model.get("num_classes", 10)),
            "--in_channels",
            str(model.get("in_channels", 3)),
            "--radial_scales",
            *[str(value) for value in radial_scales],
            "--theta_grid",
            *[str(value) for value in theta_grid],
            "--angular_random_dirs",
            str(search.get("angular_random_dirs", 4)),
            "--jacobian_eps_rel",
            str(search.get("jacobian_eps_rel", 1e-3)),
            "--jacobian_tangent_dirs",
            str(search.get("jacobian_tangent_dirs", 8)),
            "--hessian_subspace_dim",
            str(search.get("hessian_subspace_dim", 128)),
            "--hessian_topk",
            str(search.get("hessian_topk", 5)),
            "--hessian_trace_probes",
            str(search.get("hessian_trace_probes", 16)),
            "--slice_grid_size",
            str(search.get("slice_grid_size", 31)),
            "--slice_scale",
            str(search.get("slice_scale", 1.0)),
        ]
        if bool(search.get("compute_accessibility", True)):
            command.append("--compute_accessibility")
        if bool(search.get("compute_hessian", True)):
            command.append("--compute_hessian")
        if bool(search.get("compute_2d_slices", True)):
            command.append("--compute_2d_slices")
        if not bool(ablation.get("download", True)):
            command.append("--no-download")
        self.run_command(name="landscape_ablation", command=command, output_dir=output_dir)

    def run(self) -> None:
        self.write_snapshots()
        self.logger.info("Suite run_id=%s run_dir=%s", self.run_id, self.run_dir)
        self.logger.info("Checkpoint=%s label=%s", self.checkpoint_path, self.label)
        self.run_heldout()
        self.run_scaling()
        self.run_landscape_ablation()
        self.update_summary()
        self.logger.info("Suite complete: summary=%s", self.summary_path)


@hydra.main(version_base=None, config_path="../../conf/big_vae_eval_suite", config_name="config")
def main(cfg: DictConfig) -> None:
    EvalSuite(_plain(cfg)).run()


if __name__ == "__main__":
    main()
