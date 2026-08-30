from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from big_vae.models import WeightQuantileVAE, build_weight_quantile_vae
from big_vae.weightclip_benchmark.manifests import sha256_file
from training.big_vae import operator_set_overfit as operator_eval
from training.big_vae import worker
from training.big_vae.checkpointing import _save_resume_state_checkpoint
from training.weightclip_benchmark.run_ae_v11_operator_set import _load_spec


SCHEMA = "weightclip_ae_v11_postmortem_bundle_v3"
FAILED_ROOT = Path(
    "/mnt/shared/weightclip_benchmark/"
    "ae_v11_exact64_four_trunk_complement_v1_1984step"
)
DEFAULT_SPEC = Path(
    "conf/weightclip_benchmark/"
    "ae_v11_exact64_four_trunk_complement_1984step.yaml"
)
SOURCE_HASHES = {
    "big_vae/models/big_weight_vae.py": "de80e2e6122ac6897d7b72beb87f7da34f6ea704f5729ec5a6c386a17f956bae",
    "big_vae/models/vae_shared.py": "9a8153f70bd675804d46e54d5059808a9bc3b9489dc19da0da9c119bbc980047",
    "big_vae/models/big_weight_vae_parts/blocks.py": "6d10965587bcade041e329b19a88f39d4882324abacda86463dd2f9eab7ac4ae",
    "big_vae/models/big_weight_vae_parts/config.py": "fe60a59fa2b0ac8d5a5b5d4d0ea5ecf5aa81301fb35719c59020caf593666ab3",
    "big_vae/models/big_weight_vae_parts/core.py": "e7f8cf470e81954ba02d344d2847d9ebefa59b04502cedcd356d52cdab04ac6f",
    "big_vae/models/big_weight_vae_parts/encoding_mixin.py": "d7f89e4fe1636aba87f945dad875fe1fbffb8285b1ec5804d2532d13ada183d5",
    "big_vae/models/big_weight_vae_parts/decoding_mixin.py": "46a923426fbe5a5575de989adaf8b1844d2b396777685d675066c12fe4c9963e",
    "big_vae/models/big_weight_vae_parts/forward_mixin.py": "86bc3346112058b092de16d895c9e74ecc5afbb31e6b1655da038ed8b0c72206",
    "big_vae/models/big_weight_vae_parts/latent_mixin.py": "9565c98a4c04ee61360406e7fb53e1ce2677ddcdf20b63739b2b5ff84cb4b132",
    "big_vae/models/big_weight_vae_parts/loss_mixin.py": "9aa15b756590a09fc2a803ff9d7f11e2b12be0ef8f3b413e071f9e9fc5a86d27",
    "big_vae/models/distribution_encoder.py": "6e9102da889c0392d78e2e17bca6acb15b6b453adc32126a234312fb3750a23e",
    "big_vae/models/patch_tokenizers.py": "d1f94f9f81e31419523d10fac1cd15c04a9747586567c23d6d992bf9ab517cc1",
    "big_vae/datasets/operator_bank.py": "192eace4777d56dff69464eef4307d3db5a0097d58b529c5a97185124b914a5a",
    "big_vae/weightclip_benchmark/runtime_factories.py": "b4233f4e31cf3f659ee9cc35e095a22a3eefc0bdc38b76284a4fa3feb79df888",
    "training/optim.py": "10a318f8c00e3411db17b3a802a9a113fab0556de7bb1573f2ab3d896dd2b4c5",
    "training/runtime.py": "b73fd6c61d29c2ccebba1aa65cccb3103bda43a517729cf0d70eb16a691939a9",
    "training/big_vae/checkpointing.py": "af700665a303c019f7b5a078bcb0c7801fd8ae4b84086cb4dc1866c1a2e139ae",
    "training/big_vae/data.py": "756b5f32d45d3b0eeb5e20ad7f61551167c31298560e89f216cc72727aaf74fc",
    "training/big_vae/data_types.py": "57121a151875eca8555555fe84728ff317d85228a386785a6a0765b7ed15ea33",
    "training/big_vae/model_config.py": "ffb2e46d004437d9b17f4d9e24ab5ec23c6a728cb5d471ca8ff0246f4c3ba0c7",
    "training/big_vae/operator_set_overfit.py": "1681585ba98f35e2f1b2a533ef0a65923b36dcb8149b8cd91fc5758b421a468c",
    "training/big_vae/source_batching.py": "3f1c87a8442cec814510dc931f9589a53cfe3e2ef32a6e933bf76e19f57f5559",
    "training/big_vae/source_pool.py": "ab9c8aa99ce27b327ae5986a1373fd91c8357c0b93caec1debc66ed7904f92df",
    "training/big_vae/source_sampling.py": "c77eeed1deafa070d701678981ba5bfcfeb9135a94bf6396858363aae1ddea5d",
    "training/big_vae/source_state.py": "34c2e4931f415c329aba3a64228db9bf2a1a984f889c6771b620ba7ccbf0e3c1",
    "training/big_vae/two_operator_overfit.py": "e0726da74da63d2fc5a31bae682d889381460950a6653e30f82ae27862adeece",
    "training/big_vae/runtime.py": "96de9bee216d38d1baaf02ac8c35dc2ebeffbf35adedf3936ce932aaf24e4c27",
    "training/big_vae/worker.py": "085a1c7bad92b6a975353382cd94cfe0c212eb7a9b9d6bbcea3cfddedd4d4b08",
    "training/weightclip_benchmark/run_ae_v11_operator_set.py": "e34c36f47a9ea99bbdbfb3d94dbcfc9496a4fd866cd8c763c4e3ea471427ce19",
    "conf/weightclip_benchmark/ae_v11_exact64_four_trunk_complement_1984step.yaml": "5a29fccf47cc3070171df2ee44ec060453bfd4d56cda9161035344cbd6a97729",
    "tests/weightclip_benchmark/test_ae_architecture_v11.py": "addaa0ba8a38ee9b76b49c92b39cad0c17870f5e1b94d8024a2b0e6ca1962a2e",
}
FAILED_ARTIFACT_HASHES = {
    "resolved_config": "6b33560960d14812291a9dcb4452dee6f1d5b59eaa3eb3f1d85f90cf2cbe8e48",
    "selection": "88edfe2faf390e496a1dff33d67d7f5881cf0c218568c19c08a177465c816142",
    "metrics": "884d80874b2e319b23488ea30623cd3dc393322c0cdbc20163c296a48f11e5ab",
    "step1": "57a7998d34f8713de475934bd975fad6cb56faca0c7f201c266ad1e67d12e4f0",
}
ARCHIVED_STEP256 = {
    "matched_mean_dir": 0.22881455603055656,
    "matched_dir_p95": 0.23113535717129707,
    "v11_complement_normalized_mse_mean": 1.001161378214211,
    "v11_residual_to_floor_rms": 0.022193379290633496,
    "v11_residual_to_floor_rms_mean": 0.025850852398434654,
    "v11_trunk_sample_structure_linear_cka_mean": 0.9935464262962341,
}
REPLAY_STEPS = 256
TILES_PER_STEP = 18
GRADIENT_PANEL_STEPS = 8
PYTHON_CLOSURE_ROOTS = ("big_vae", "training", "dataset", "experiments")
PYTHON_CLOSURE_SHA256 = (
    "40678fde6cf361e075e4fc8ada692fe22c3c7ebe33c1ee2242b51115d2bd17bb"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay V11 to post-update step 256, then run a zero-update "
            "objective/batch-gradient panel and exact64 cross-fit ridge probe."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _model_state_digest(model: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    total_numel = 0
    inventory = [
        *(('parameter', name, tensor) for name, tensor in model.named_parameters()),
        *(('buffer', name, tensor) for name, tensor in model.named_buffers()),
    ]
    for kind, name, tensor in inventory:
        detached = tensor.detach().contiguous().cpu()
        header = {
            "kind": kind,
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
        }
        digest.update(_canonical_json(header))
        digest.update(memoryview(detached.reshape(-1).view(torch.uint8).numpy()))
        tensor_count += 1
        total_numel += detached.numel()
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "total_numel": total_numel,
        "coverage": "all named parameters plus all named persistent/nonpersistent buffers",
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _default_output_root() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(
        "/mnt/shared/weightclip_benchmark/"
        f"ae_v11_postmortem_bundle_v3_{stamp}"
    )


def _source_binding(workspace: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = workspace / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen V11 source drift: {relative} expected={expected} "
                f"actual={actual}"
            )
        rows[relative] = {"sha256": actual}
    analyzer = Path(__file__).resolve()
    closure_rows: list[dict[str, str]] = []
    for root_name in PYTHON_CLOSURE_ROOTS:
        for path in sorted((workspace / root_name).rglob("*.py")):
            resolved = path.resolve()
            if resolved == analyzer:
                continue
            closure_rows.append(
                {
                    "path": str(resolved.relative_to(workspace)),
                    "sha256": sha256_file(resolved),
                }
            )
    closure_sha256 = hashlib.sha256(_canonical_json(closure_rows)).hexdigest()
    if closure_sha256 != PYTHON_CLOSURE_SHA256:
        raise RuntimeError(
            "frozen repo-local Python closure drift: "
            f"expected={PYTHON_CLOSURE_SHA256} actual={closure_sha256}"
        )
    rows["__repo_local_python_closure__"] = {
        "roots": list(PYTHON_CLOSURE_ROOTS),
        "excluded": str(analyzer.relative_to(workspace)),
        "file_count": len(closure_rows),
        "aggregate_sha256": closure_sha256,
        "files": closure_rows,
    }
    return rows


def _validate_failed_run_binding() -> dict[str, Any]:
    required = {
        "resolved_config": FAILED_ROOT / "resolved_run_config.json",
        "selection": FAILED_ROOT / "operator_set_selection.json",
        "metrics": FAILED_ROOT / "operator_set_metrics.jsonl",
        "step1": FAILED_ROOT / "v11_exact_b18_step1_gradients_v1.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"failed V11 binding missing {name}: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != FAILED_ARTIFACT_HASHES[name]:
            raise RuntimeError(
                f"failed V11 artifact drift: {name} expected={FAILED_ARTIFACT_HASHES[name]} "
                f"actual={actual_hash}"
            )
    rows = [
        json.loads(line)
        for line in required["metrics"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(float(row["step"])) for row in rows] != [0, 256, 512]:
        raise RuntimeError("failed V11 metric ledger must contain exact steps 0,256,512")
    archived = rows[1]
    for name, expected in ARCHIVED_STEP256.items():
        actual = float(archived[name])
        if actual != expected:
            raise RuntimeError(
                f"archived step256 signature drift: {name}={actual} expected={expected}"
            )
    return {
        "root": str(FAILED_ROOT),
        "files": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
        "archived_step256": {name: archived[name] for name in ARCHIVED_STEP256},
        "failed_step512": {
            "matched_mean_dir": rows[2]["matched_mean_dir"],
            "v11_floor_mean_dir": rows[2]["v11_floor_mean_dir"],
            "v11_complement_normalized_mse_mean": rows[2][
                "v11_complement_normalized_mse_mean"
            ],
        },
    }


def _rebase_worker_config(
    cfg: dict[str, Any],
    *,
    output_root: Path,
    ephemeral_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    def rebase_failed_path(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rebase_failed_path(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase_failed_path(item) for item in value]
        if isinstance(value, str) and str(FAILED_ROOT) in value:
            relative = value.split(str(FAILED_ROOT), 1)[1].lstrip("/")
            return str(ephemeral_root / "rebased_failed_paths" / relative)
        return value

    cfg = rebase_failed_path(cfg)
    run_id = "weightclip_ae_v11_postmortem_replay256_v3"
    run_root = output_root / "replay" / "runs" / run_id
    log_dir = run_root / "logs"
    ephemeral_checkpoint = ephemeral_root / "checkpoints" / run_id
    cfg["run_config"]["name"] = run_id
    cfg["logging"].update(
        {
            "dir": str(log_dir),
            "file_path": str(log_dir / "train_rank0.log"),
            "file_name": "train_rank0.log",
        }
    )
    cfg["training_artifacts"].update(
        {
            "base_root_dir": str(output_root / "replay"),
            "root_dir": str(output_root / "replay"),
            "run_id": run_id,
            "run_root_dir": str(run_root),
            "runs_dir": str(output_root / "replay" / "runs"),
            "logs_dir": str(log_dir),
            "reports_dir": str(run_root / "reports"),
            "crashes_dir": str(ephemeral_root / "crashes"),
            "checkpoints_dir": str(ephemeral_root / "checkpoints"),
            "big_vae_checkpoint_dir": str(ephemeral_checkpoint),
            "tmp_dir": str(ephemeral_root / "tmp"),
        }
    )
    cfg["collector"]["diagnostics"].update(
        {
            "crash_report_path": str(ephemeral_root / "collector_crash.json"),
            "worker_status_dir": str(ephemeral_root / "worker_status"),
        }
    )
    train = cfg["train"]
    train["checkpoint_dir"] = str(ephemeral_checkpoint)
    train["operator_bank"]["operator_set_overfit"]["metrics_path"] = str(
        output_root / "replay" / "operator_set_metrics.jsonl"
    )
    train["telemetry"]["comet"]["enabled"] = False
    train["telemetry"]["wandb"]["enabled"] = False
    stage_dir = ephemeral_checkpoint / "stage_1"
    train["fixed_training_batch"]["dump_path"] = str(
        stage_dir / "fixed_training_batch.pt"
    )
    grad = train["telemetry"]["grad_layer_monitor"]
    grad.update(
        {
            "csv_path": str(output_root / "replay" / "grad_layer_rms.csv"),
            "plot_path": str(ephemeral_root / "grad_layer_rms.png"),
            "heatmap_path": str(ephemeral_root / "grad_layer_rms_heatmap.png"),
            "save_plot": False,
            "save_heatmap": False,
        }
    )
    leaked: list[str] = []

    def find_leaks(value: Any, prefix: str = "cfg") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                find_leaks(item, f"{prefix}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                find_leaks(item, f"{prefix}[{index}]")
        elif isinstance(value, str) and str(FAILED_ROOT) in value:
            leaked.append(f"{prefix}={value}")

    find_leaks(cfg)
    if leaked:
        raise RuntimeError(f"replay config can mutate failed V11 root: {leaked}")
    return cfg


def _signature_deltas(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected in ARCHIVED_STEP256.items():
        actual = float(row[name])
        absolute = abs(actual - expected)
        actual_10dp = f"{actual:.10f}"
        archived_10dp = f"{expected:.10f}"
        result[name] = {
            "actual": actual,
            "archived": expected,
            "absolute_delta": absolute,
            "actual_10dp": actual_10dp,
            "archived_10dp": archived_10dp,
            "gate": "exact equality after fixed 10-decimal formatting",
            "passed": actual_10dp == archived_10dp,
        }
    return result


@contextlib.contextmanager
def _bounded_replay_hooks(
    *,
    cfg: dict[str, Any],
    output_root: Path,
    resume_dir: Path,
) -> Iterator[dict[str, Any]]:
    registry: dict[str, Any] = {"range_hits": 0, "captured": False}
    original_range = getattr(worker, "range", None)
    had_range = hasattr(worker, "range")
    original_optimizer = worker._build_optimizer
    original_scheduler = worker._build_scheduler
    original_scaler = worker.runtime_create_grad_scaler
    original_evaluate = operator_eval.OperatorSetOverfitEvaluator.evaluate

    def bounded_range(*args: int) -> range:
        if tuple(int(value) for value in args) == (0, 1984):
            registry["range_hits"] += 1
            return builtins.range(0, REPLAY_STEPS)
        return builtins.range(*args)

    def capture_optimizer(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        value = original_optimizer(*args, **kwargs)
        registry["optimizer"] = value
        return value

    def capture_scheduler(*args: Any, **kwargs: Any) -> Any:
        value = original_scheduler(*args, **kwargs)
        registry["scheduler"] = value
        return value

    def capture_scaler(*args: Any, **kwargs: Any) -> Any:
        value = original_scaler(*args, **kwargs)
        registry["scaler"] = value
        return value

    def capture_evaluate(
        evaluator: operator_eval.OperatorSetOverfitEvaluator,
        *,
        model: torch.nn.Module,
        step: int,
        microbatches: Sequence[tuple[torch.Tensor, ...]] | None = None,
        eval_batch_size: int = 8,
    ) -> dict[str, Any]:
        row = original_evaluate(
            evaluator,
            model=model,
            step=step,
            microbatches=microbatches,
            eval_batch_size=eval_batch_size,
        )
        if int(step) != REPLAY_STEPS:
            return row
        operator_eval.validate_v11_geometry_metrics(row)
        operator_eval.validate_v11_numeric_telemetry_finite(row)
        deltas = _signature_deltas(row)
        failures = [name for name, item in deltas.items() if not item["passed"]]
        if failures:
            raise RuntimeError(f"replay step256 signature mismatch: {failures}")
        missing = [
            name
            for name in ("optimizer", "scheduler", "scaler")
            if name not in registry
        ]
        if missing:
            raise RuntimeError(f"replay snapshot missing runtime state: {missing}")
        active_device = next(model.parameters()).device
        _save_resume_state_checkpoint(
            model=model,
            optimizer=registry["optimizer"],
            scheduler=registry["scheduler"],
            scaler=registry["scaler"],
            cfg=OmegaConf.create(cfg),
            step_idx=REPLAY_STEPS,
            logger=worker._logger("v11_postmortem_snapshot", rank=0),
            stage=1,
            state_dir=resume_dir,
            rank=0,
            world_size=1,
            active_device=active_device,
        )
        registry["captured"] = True
        registry["signature"] = deltas
        _atomic_json(
            output_root / "replay" / "signature_gate.json",
            {
                "schema": SCHEMA,
                "step": REPLAY_STEPS,
                "post_update": True,
                "committed_logical_index": REPLAY_STEPS * TILES_PER_STEP,
                "metrics": deltas,
                "geometry_pass": True,
            },
        )
        return row

    worker.range = bounded_range
    worker._build_optimizer = capture_optimizer
    worker._build_scheduler = capture_scheduler
    worker.runtime_create_grad_scaler = capture_scaler
    operator_eval.OperatorSetOverfitEvaluator.evaluate = capture_evaluate
    try:
        yield registry
    finally:
        operator_eval.OperatorSetOverfitEvaluator.evaluate = original_evaluate
        worker.runtime_create_grad_scaler = original_scaler
        worker._build_scheduler = original_scheduler
        worker._build_optimizer = original_optimizer
        if had_range:
            worker.range = original_range
        else:
            delattr(worker, "range")


def _run_replay(
    *, cfg: dict[str, Any], output_root: Path, ephemeral_root: Path
) -> tuple[Path, dict[str, Any]]:
    print("[v11-postmortem] stage=replay256 start", flush=True)
    resume_dir = ephemeral_root / "resume_state"
    with _bounded_replay_hooks(
        cfg=cfg, output_root=output_root, resume_dir=resume_dir
    ) as registry:
        worker._run_worker(
            rank=0,
            world_size=1,
            cfg_dict=cfg,
            master_addr="127.0.0.1",
            master_port=0,
            monitor_queue=None,
        )
    resume_path = resume_dir / "step_0000256.pt"
    if not registry.get("captured") or not resume_path.is_file():
        raise RuntimeError("bounded replay did not persist the ephemeral step256 state")
    if int(registry.get("range_hits", 0)) < 2:
        raise RuntimeError(
            "bounded replay did not intercept both training and prefetch horizons"
        )
    result = {
        "range_hits": int(registry["range_hits"]),
        "resume_path": str(resume_path),
        "resume_size": resume_path.stat().st_size,
        "ephemeral": True,
        "signature": registry["signature"],
    }
    print(
        "[v11-postmortem] stage=replay256 complete "
        f"cursor={REPLAY_STEPS * TILES_PER_STEP} resume={resume_path}",
        flush=True,
    )
    return resume_path, result


def _build_source_context(
    cfg: Mapping[str, Any],
) -> contextlib.AbstractContextManager[Any]:
    operator = cfg["train"]["operator_bank"]
    exact = operator["operator_set_overfit"]
    return operator_eval.canonical_operator_set_data_pipeline(
        operator["pair_manifest"],
        selected_operators=exact["selected_operators"],
        expected_operator_count=64,
        seed=int(cfg["data"]["seed"]),
        hot_shards=int(operator["hot_shards"]),
        expected_pair_manifest_sha256=str(operator["pair_manifest_sha256"]),
        expected_selection_sha256=str(exact["selection_sha256"]),
        expected_schedule_sha256=str(exact["schedule_sha256"]),
        max_active_strata=int(operator["max_active_strata"]),
        max_active_bundle_bytes=int(operator["max_active_bundle_bytes"]),
    )


def _load_snapshot_model(
    *, cfg: dict[str, Any], resume_path: Path
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    torch.device,
    bool,
    torch.dtype | None,
    dict[str, Any],
]:
    device = worker._resolve_device(OmegaConf.create(cfg), rank=0, world_size=1)
    if device.type != "cuda":
        raise RuntimeError("V11 postmortem requires the same CUDA production backend")
    worker._set_speed_optimizations(OmegaConf.create(cfg), device=device)
    worker._enforce_v10_strict_fp32_backend(OmegaConf.create(cfg), device=device)
    worker._seed_everything(int(cfg["data"]["seed"]), active_cuda_device=device)
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != REPLAY_STEPS:
        raise RuntimeError("ephemeral resume payload is not post-update step256")
    data_stream = payload.get("rng_state", {}).get("data_stream", {})
    expected_cursor = REPLAY_STEPS * TILES_PER_STEP
    if (
        not bool(data_stream.get("exactly_restorable", False))
        or int(data_stream.get("committed_training_step", -1)) != REPLAY_STEPS
        or int(data_stream.get("logical_sample_index", -1)) != expected_cursor
    ):
        raise RuntimeError(f"ephemeral resume cursor mismatch: {data_stream}")
    cfg_omega = OmegaConf.create(cfg)
    model = build_weight_quantile_vae(worker._build_model_cfg(cfg_omega)).to(device)
    normalized = worker._normalize_model_state_dict_keys(payload["model_state"])
    missing, unexpected = model.load_state_dict(normalized, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"snapshot state mismatch missing={missing} unexpected={unexpected}"
        )
    amp_enabled, amp_dtype = worker._resolve_amp(cfg=cfg_omega, device=device)
    optimizer = worker._build_optimizer(model, cfg_omega, device)
    name_by_identity = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    optimizer_names = [
        [name_by_identity[id(parameter)] for parameter in group["params"]]
        for group in optimizer.param_groups
    ]
    if optimizer_names != payload.get("optimizer_param_names"):
        raise RuntimeError("reloaded optimizer parameter identity/order mismatch")
    optimizer.load_state_dict(payload["optimizer_state"])
    loaded_optimizer = optimizer.state_dict()
    scheduler = worker._build_scheduler(optimizer, cfg_omega)
    if payload.get("scheduler_state") is None:
        if scheduler is not None:
            raise RuntimeError("snapshot lacks state for configured scheduler")
    else:
        if scheduler is None:
            raise RuntimeError("snapshot has scheduler state but scheduler is disabled")
        scheduler.load_state_dict(payload["scheduler_state"])
        if scheduler.state_dict() != payload["scheduler_state"]:
            raise RuntimeError("reloaded scheduler state differs from snapshot")
    scaler = worker.runtime_create_grad_scaler(
        device=device,
        enabled=(amp_enabled and amp_dtype == torch.float16),
    )
    scaler.load_state_dict(payload["scaler_state"])
    if scaler.state_dict() != payload["scaler_state"]:
        raise RuntimeError("reloaded scaler state differs from snapshot")
    runtime_restore = {
        "model_state": "strict load_state_dict",
        "optimizer_state_entries": len(loaded_optimizer["state"]),
        "optimizer_parameter_groups": len(loaded_optimizer["param_groups"]),
        "optimizer_parameter_order_exact": True,
        "scheduler_state_exact": True,
        "scaler_state_exact": True,
        "rng_state": "Python/NumPy/CPU/active-CUDA restored after runtime rebuild",
    }
    del optimizer, scheduler, scaler, loaded_optimizer
    torch.cuda.empty_cache()
    worker._restore_training_rng_state(
        payload["rng_state"], active_cuda_device=device
    )
    model.eval()
    return model, payload, device, amp_enabled, amp_dtype, runtime_restore


def _compare_rows(
    left: Mapping[str, Any], right: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        l_value = float(left[name])
        r_value = float(right[name])
        delta = abs(l_value - r_value)
        tolerance = max(1.0e-7, 1.0e-6 * max(abs(l_value), abs(r_value)))
        result[name] = {
            "left": l_value,
            "right": r_value,
            "absolute_delta": delta,
            "tolerance": tolerance,
            "passed": delta <= tolerance,
        }
    return result


def _reload_parity_eval(
    *,
    model: torch.nn.Module,
    source: Any,
    output_root: Path,
    archived_row: Mapping[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[operator_eval.OperatorSetOverfitEvaluator, dict[str, Any]]:
    print("[v11-postmortem] stage=reload-parity start", flush=True)
    evaluator = operator_eval.OperatorSetOverfitEvaluator(
        source=source,
        output_path=output_root / "analysis" / "reload_metrics.jsonl",
        every_steps=256,
        patch_size=16,
        gamma=0.5,
        lambda_dir=1.0,
        lambda_scale=0.1,
        huber_delta=0.1,
    )
    with worker._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        row = evaluator.evaluate(model=model, step=REPLAY_STEPS, eval_batch_size=8)
    operator_eval.validate_v11_geometry_metrics(row)
    operator_eval.validate_v11_numeric_telemetry_finite(row)
    names = tuple(ARCHIVED_STEP256) + (
        "v11_basis_orthogonality_max_abs",
        "v11_complement_input_rowspace_max_abs",
        "v11_native_latent_capture_max_abs",
        "v11_manual_decode_max_abs",
    )
    comparisons = _compare_rows(archived_row, row, names)
    failures = [name for name, item in comparisons.items() if not item["passed"]]
    if failures:
        raise RuntimeError(f"reloaded step256 parity mismatch: {failures}")
    result = {
        "schema": SCHEMA,
        "step": REPLAY_STEPS,
        "geometry_pass": True,
        "nonfinite_pass": True,
        "comparisons": comparisons,
    }
    _atomic_json(output_root / "analysis" / "reload_parity.json", result)
    print("[v11-postmortem] stage=reload-parity complete", flush=True)
    return evaluator, row


def _capture_ridge_table(
    *,
    model: torch.nn.Module,
    evaluator: operator_eval.OperatorSetOverfitEvaluator,
    source: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    raw_model = operator_eval._raw_model(model)
    encoder = raw_model.four_trunk_complement_encoder_v11
    if encoder is None or len(encoder.trunks) != 4:
        raise RuntimeError("ridge capture requires the exact four-trunk V11 encoder")
    assignments = evaluator._evaluation_assignments()
    microbatches = evaluator._materialize_evaluation_microbatches(
        device=device, batch_size=8
    )
    codes: list[list[torch.Tensor]] = [[] for _ in range(4)]
    complements: list[torch.Tensor] = []
    latent_parity_max = 0.0
    was_training = raw_model.training
    raw_model.eval()
    encoder.capture_trunk_states = True
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(microbatches):
                W_cpu, x_cpu, x_mask_cpu, d_in_cpu, d_out_cpu, _indices = batch
                W = W_cpu.to(device)
                x = x_cpu.to(device)
                x_mask = x_mask_cpu.to(device)
                d_in_mask = d_in_cpu.to(device)
                d_out_mask = d_out_cpu.to(device)
                with worker._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    _W_hat, _mu, _logvar, _pred, debug = raw_model.forward_debug(
                        W,
                        x,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                        disable_z_shortcut=True,
                    )
                if (
                    len(encoder.last_trunk_codes) != 4
                    or encoder.last_protected is None
                    or encoder.last_complement_patches is None
                ):
                    raise RuntimeError(f"incomplete ridge capture at batch {batch_index}")
                for trunk, code in enumerate(encoder.last_trunk_codes):
                    codes[trunk].append(code.detach().float().cpu())
                complements.append(
                    encoder.last_complement_patches.detach().float().cpu()
                )
                captured = torch.cat(
                    [encoder.last_protected, *encoder.last_trunk_codes], dim=-1
                ).reshape_as(debug["latent_decoder_z"])
                latent_parity_max = max(
                    latent_parity_max,
                    float(
                        (captured.float() - debug["latent_decoder_z"].float())
                        .abs()
                        .amax()
                        .item()
                    ),
                )
    finally:
        operator_eval._reset_v11_evaluator_capture(encoder)
        raw_model.train(was_training)
    if latent_parity_max != 0.0:
        raise RuntimeError(f"ridge latent capture parity is not exact: {latent_parity_max}")
    feature_rows = [torch.cat(chunks, dim=0).flatten(1) for chunks in codes]
    target = torch.cat(complements, dim=0).flatten(1)
    if tuple(target.shape) != (576, 128 * 8 * 16):
        raise RuntimeError(f"ridge complement orientation drift: {tuple(target.shape)}")
    if any(tuple(feature.shape) != (576, 32 * 16) for feature in feature_rows):
        raise RuntimeError("ridge per-trunk feature shape drift")
    if not all(bool(torch.isfinite(value).all()) for value in [*feature_rows, target]):
        raise RuntimeError("ridge capture contains nonfinite values")
    operator_indices: list[int] = []
    tile_indices: list[int] = []
    datasets: list[str] = []
    operator_lookup = {key: index for index, key in enumerate(source._keys)}
    for key, tile_index in assignments:
        operator_indices.append(operator_lookup[key])
        tile_indices.append(int(tile_index))
        datasets.append(str(source.operator_groups[key][0].metadata["dataset"]))
    tiles_by_operator: dict[int, set[int]] = {}
    for operator, tile in zip(operator_indices, tile_indices, strict=True):
        tiles_by_operator.setdefault(operator, set()).add(tile)
    if len(tiles_by_operator) != 64 or any(
        tiles != set(range(9)) for tiles in tiles_by_operator.values()
    ):
        raise RuntimeError(
            "ridge evaluation table must contain every tile 0..8 exactly once "
            "for each of the 64 operators"
        )
    return {
        "features": {
            **{f"trunk_{index:02d}": value for index, value in enumerate(feature_rows)},
            "concat_za": torch.cat(feature_rows, dim=1),
        },
        "target": target,
        "operator_indices": torch.tensor(operator_indices, dtype=torch.long),
        "tile_indices": torch.tensor(tile_indices, dtype=torch.long),
        "datasets": datasets,
        "assignments": assignments,
        "latent_parity_max_abs": latent_parity_max,
    }


def _operator_split(table: Mapping[str, Any]) -> dict[str, Any]:
    operator_indices = table["operator_indices"]
    datasets = table["datasets"]
    dataset_by_operator: dict[int, str] = {}
    for operator, dataset in zip(operator_indices.tolist(), datasets, strict=True):
        existing = dataset_by_operator.setdefault(int(operator), str(dataset))
        if existing != dataset:
            raise RuntimeError("one operator appears under multiple dataset families")
    by_dataset: dict[str, list[int]] = {}
    for operator, dataset in dataset_by_operator.items():
        by_dataset.setdefault(dataset, []).append(operator)
    if len(dataset_by_operator) != 64 or len(by_dataset) != 10:
        raise RuntimeError("ridge split requires exact 64 operators across 10 datasets")
    base_counts = {dataset: len(operators) // 4 for dataset, operators in by_dataset.items()}
    remaining = 16 - sum(base_counts.values())
    ranked_datasets = sorted(
        by_dataset,
        key=lambda dataset: (
            -(len(by_dataset[dataset]) / 4.0 - base_counts[dataset]),
            hashlib.sha256(f"42|{dataset}".encode()).hexdigest(),
        ),
    )
    for dataset in ranked_datasets[:remaining]:
        base_counts[dataset] += 1
    test_operators: list[int] = []
    for dataset, operators in sorted(by_dataset.items()):
        ordered = sorted(
            operators,
            key=lambda operator: hashlib.sha256(
                f"42|test|{dataset}|{operator}".encode()
            ).hexdigest(),
        )
        test_operators.extend(ordered[: base_counts[dataset]])
    test_set = set(test_operators)
    train_operators = sorted(set(dataset_by_operator) - test_set)
    if len(train_operators) != 48 or len(test_set) != 16:
        raise RuntimeError("ridge grouped split is not exact 48/16")
    validation_operators = sorted(
        train_operators,
        key=lambda operator: hashlib.sha256(
            f"42|validation|{dataset_by_operator[operator]}|{operator}".encode()
        ).hexdigest(),
    )[:8]
    fit_operators = sorted(set(train_operators) - set(validation_operators))
    row_operator = operator_indices.tolist()
    masks = {
        "fit": torch.tensor(
            [int(value) in set(fit_operators) for value in row_operator],
            dtype=torch.bool,
        ),
        "validation": torch.tensor(
            [int(value) in set(validation_operators) for value in row_operator],
            dtype=torch.bool,
        ),
        "train": torch.tensor(
            [int(value) in set(train_operators) for value in row_operator],
            dtype=torch.bool,
        ),
        "test": torch.tensor(
            [int(value) in test_set for value in row_operator], dtype=torch.bool
        ),
    }
    expected_rows = {"fit": 360, "validation": 72, "train": 432, "test": 144}
    for name, expected in expected_rows.items():
        if int(masks[name].sum()) != expected:
            raise RuntimeError(
                f"ridge {name} row count drift: {int(masks[name].sum())} != {expected}"
            )
    return {
        "dataset_by_operator": dataset_by_operator,
        "fit_operators": fit_operators,
        "validation_operators": validation_operators,
        "train_operators": train_operators,
        "test_operators": sorted(test_set),
        "masks": masks,
    }


def _shuffle_features_within_partition(
    features: torch.Tensor,
    *,
    operator_indices: torch.Tensor,
    tile_indices: torch.Tensor,
    partitions: Sequence[Sequence[int]],
) -> torch.Tensor:
    result = torch.empty_like(features)
    assigned = torch.zeros(features.shape[0], dtype=torch.bool)
    lookup = {
        (int(operator), int(tile)): row
        for row, (operator, tile) in enumerate(
            zip(operator_indices.tolist(), tile_indices.tolist(), strict=True)
        )
    }
    for raw_partition in partitions:
        partition = sorted(int(value) for value in raw_partition)
        if len(partition) < 2:
            raise RuntimeError("ridge shuffled control partition is degenerate")
        rotated = partition[1:] + partition[:1]
        for destination_operator, source_operator in zip(
            partition, rotated, strict=True
        ):
            for tile in range(9):
                destination = lookup[(destination_operator, tile)]
                source = lookup[(source_operator, tile)]
                result[destination] = features[source]
                assigned[destination] = True
    if not bool(assigned.all()):
        raise RuntimeError("ridge shuffled control omitted rows")
    return result


def _normalized_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> float:
    numerator = (prediction.float() - target.float()).square().sum(dtype=torch.float64)
    denominator = target.float().square().sum(dtype=torch.float64).clamp_min(1.0e-30)
    return float((numerator / denominator).item())


def _dual_ridge_prediction(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    *,
    relative_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    x_train = torch.cat(
        [x_train, torch.ones(x_train.shape[0], 1, device=x_train.device)], dim=1
    )
    x_test = torch.cat(
        [x_test, torch.ones(x_test.shape[0], 1, device=x_test.device)], dim=1
    )
    gram = x_train @ x_train.transpose(0, 1)
    gram_scale = float(gram.diagonal().mean().item())
    ridge = float(relative_lambda) * max(gram_scale, 1.0e-12)
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(regularized, train_y)
    prediction = (x_test @ x_train.transpose(0, 1)) @ coefficients
    singular = torch.linalg.eigvalsh(gram.double()).clamp_min(0.0)
    threshold = max(float(singular.max().item()) * 1.0e-10, 1.0e-12)
    rank = int((singular > threshold).sum().item())
    condition = float(
        ((singular.max() + ridge) / (singular.min() + ridge)).item()
    )
    return prediction, {
        "gram_scale": gram_scale,
        "absolute_lambda": ridge,
        "gram_rank": rank,
        "regularized_condition": condition,
    }


def _ridge_feature(
    *,
    name: str,
    features: torch.Tensor,
    target: torch.Tensor,
    split: Mapping[str, Any],
    operator_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    masks = {
        name: value.to(device=device) for name, value in split["masks"].items()
    }
    x = features.to(device=device, dtype=torch.float32)
    y = target.to(device=device, dtype=torch.float32)
    lambdas = (1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
    validation: list[dict[str, Any]] = []
    for relative_lambda in lambdas:
        prediction, geometry = _dual_ridge_prediction(
            x[masks["fit"]],
            y[masks["fit"]],
            x[masks["validation"]],
            relative_lambda=relative_lambda,
        )
        validation.append(
            {
                "relative_lambda": relative_lambda,
                "normalized_mse": _normalized_mse(
                    prediction, y[masks["validation"]]
                ),
                **geometry,
            }
        )
    best = min(validation, key=lambda row: row["normalized_mse"])
    test_prediction, geometry = _dual_ridge_prediction(
        x[masks["train"]],
        y[masks["train"]],
        x[masks["test"]],
        relative_lambda=float(best["relative_lambda"]),
    )
    test_target = y[masks["test"]]
    test_ops = operator_indices.to(device)[masks["test"]]
    per_operator = {}
    for operator in split["test_operators"]:
        mask = test_ops == int(operator)
        per_operator[str(operator)] = _normalized_mse(
            test_prediction[mask], test_target[mask]
        )
    return {
        "feature": name,
        "feature_dim": int(features.shape[1]),
        "validation_grid": validation,
        "selected_relative_lambda": best["relative_lambda"],
        "test_normalized_mse": _normalized_mse(test_prediction, test_target),
        "test_r2_zero_baseline": 1.0
        - _normalized_mse(test_prediction, test_target),
        "test_normalized_mse_by_operator": per_operator,
        **geometry,
    }


def _run_ridge(
    *, table: Mapping[str, Any], output_root: Path, device: torch.device
) -> dict[str, Any]:
    print("[v11-postmortem] stage=ridge start split=48/16", flush=True)
    split = _operator_split(table)
    features = dict(table["features"])
    concat = features["concat_za"]
    trunk_features = [features[f"trunk_{index:02d}"] for index in range(4)]
    for held_out in range(4):
        features[f"concat_without_trunk_{held_out:02d}"] = torch.cat(
            [
                value
                for index, value in enumerate(trunk_features)
                if index != held_out
            ],
            dim=1,
        )
    features["concat_za_shuffled"] = _shuffle_features_within_partition(
        concat,
        operator_indices=table["operator_indices"],
        tile_indices=table["tile_indices"],
        partitions=(split["train_operators"], split["test_operators"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(11_256)
    gaussian = torch.randn(concat.shape, generator=generator)
    features["concat_gaussian_diag_matched"] = (
        gaussian
        * concat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        + concat.mean(dim=0, keepdim=True)
    )
    rows = []
    for name, value in features.items():
        print(f"[v11-postmortem] stage=ridge feature={name}", flush=True)
        row = _ridge_feature(
            name=name,
            features=value,
            target=table["target"],
            split=split,
            operator_indices=table["operator_indices"],
            device=device,
        )
        rows.append(row)
        _atomic_json(
            output_root / "analysis" / "ridge_partial.json",
            {"schema": SCHEMA, "complete": False, "features": rows},
        )
    result = {
        "schema": SCHEMA,
        "target": "exact captured P_perp(W) in [d_out,T,p] orientation",
        "rows": 576,
        "target_dim": int(table["target"].shape[1]),
        "split": {
            key: value
            for key, value in split.items()
            if key != "masks" and key != "dataset_by_operator"
        },
        "dataset_by_operator": {
            str(key): value for key, value in split["dataset_by_operator"].items()
        },
        "latent_capture_max_abs": table["latent_parity_max_abs"],
        "features": rows,
    }
    _atomic_json(output_root / "analysis" / "ridge_report.json", result)
    print("[v11-postmortem] stage=ridge complete", flush=True)
    return result


def _logical_assignment(source: Any, logical_index: int) -> tuple[Any, int]:
    records = int(source._records_per_cycle)
    cycle, offset = divmod(int(logical_index), records)
    return source.locality_plan(cycle)[offset]


def _materialize_logical_microbatch(
    source: Any,
    logical_indices: Sequence[int],
    group_cache: dict[tuple[Any, ...], Sequence[Any]],
) -> tuple[torch.Tensor, ...]:
    rows = []
    operator_indices = []
    tile_indices = []
    key_lookup = {key: index for index, key in enumerate(source._keys)}
    for logical_index in logical_indices:
        key, tile_index = _logical_assignment(source, int(logical_index))
        if key not in group_cache:
            group_cache[key] = source._materialize_group(key, None)
        rows.append(group_cache[key][int(tile_index)])
        operator_indices.append(key_lookup[key])
        tile_indices.append(int(tile_index))
    return (
        torch.stack([row.weight for row in rows]),
        torch.stack([row.x for row in rows]),
        torch.stack([row.meta["x_mask"] for row in rows]),
        torch.stack([row.meta["d_in_mask"] for row in rows]),
        torch.stack([row.meta["d_out_mask"] for row in rows]),
        torch.tensor(operator_indices, dtype=torch.long),
        torch.tensor(tile_indices, dtype=torch.long),
    )


def _gradient_groups(
    model: torch.nn.Module,
) -> tuple[list[str], list[torch.nn.Parameter], dict[str, list[int]]]:
    inventory = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _parameter in inventory]
    parameters = [parameter for _name, parameter in inventory]
    groups: dict[str, list[int]] = {"all_trainable": list(range(len(parameters)))}

    def add(name: str, prefixes: Sequence[str]) -> None:
        indices = [
            index
            for index, parameter_name in enumerate(names)
            if parameter_name.startswith(tuple(prefixes))
        ]
        if not indices:
            raise RuntimeError(f"gradient group {name} has no parameters")
        groups[name] = indices

    add("distribution_encoder", ("distribution_encoder.",))
    for trunk in range(4):
        prefix = f"four_trunk_complement_encoder_v11.trunks.{trunk}."
        add(f"trunk_{trunk:02d}", (prefix,))
        add(f"code_projection_{trunk:02d}", (prefix + "code_projection.",))
    add("mandatory_latent_bridge", ("mandatory_latent_bridge.",))
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
        add(
            f"mandatory_latent_bridge_{projection}",
            (f"mandatory_latent_bridge.{projection}.",),
        )
    add("decoder_layers", ("decoder_layers.", "decoder_post_norms."))
    add("v11_residual_head", ("v11_residual_head.",))
    if len(parameters) != 474 or sum(parameter.numel() for parameter in parameters) != 235_073_152:
        raise RuntimeError("gradient panel trainable topology is not frozen V11")
    return names, parameters, groups


def _select_gradient(
    gradients: Sequence[torch.Tensor | None], indices: Sequence[int]
) -> tuple[torch.Tensor | None, ...]:
    return tuple(gradients[index] for index in indices)


def _gradient_dot(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> float:
    first = next(
        (
            tensor
            for pair in zip(left, right, strict=True)
            for tensor in pair
            if tensor is not None
        ),
        None,
    )
    if first is None:
        return 0.0
    value = torch.zeros((), device=first.device)
    for left_tensor, right_tensor in zip(left, right, strict=True):
        if left_tensor is None or right_tensor is None:
            continue
        value = value + (left_tensor.float() * right_tensor.float()).sum()
    return float(value.item())


def _gradient_norm(gradient: Sequence[torch.Tensor | None]) -> float:
    return math.sqrt(max(_gradient_dot(gradient, gradient), 0.0))


def _gradient_add(
    gradients: Sequence[Sequence[torch.Tensor | None]],
    weights: Sequence[float],
) -> tuple[torch.Tensor | None, ...]:
    if len(gradients) != len(weights) or not gradients:
        raise ValueError("gradient weighted sum shape mismatch")
    result: list[torch.Tensor | None] = []
    for tensors in zip(*gradients, strict=True):
        present = [
            float(weight) * tensor.float()
            for weight, tensor in zip(weights, tensors, strict=True)
            if tensor is not None
        ]
        result.append(sum(present[1:], present[0]) if present else None)
    return tuple(result)


def _cosine(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> float | None:
    left_norm = _gradient_norm(left)
    right_norm = _gradient_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return _gradient_dot(left, right) / (left_norm * right_norm)


def _gradient_gram(
    gradients: Sequence[Sequence[torch.Tensor | None]],
) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradient Gram requires at least one gradient")
    width = len(gradients[0])
    if any(len(gradient) != width for gradient in gradients):
        raise ValueError("gradient Gram tensor inventories differ")
    first = next(
        (
            tensor
            for gradient in gradients
            for tensor in gradient
            if tensor is not None
        ),
        None,
    )
    if first is None:
        return torch.zeros((len(gradients), len(gradients)), dtype=torch.float64)
    gram = torch.zeros(
        (len(gradients), len(gradients)),
        device=first.device,
        dtype=torch.float32,
    )
    for tensors in zip(*gradients, strict=True):
        reference = next((tensor for tensor in tensors if tensor is not None), None)
        if reference is None:
            continue
        flat = torch.stack(
            [
                torch.zeros_like(reference, dtype=torch.float32).flatten()
                if tensor is None
                else tensor.float().flatten()
                for tensor in tensors
            ],
            dim=0,
        )
        gram.add_(flat @ flat.transpose(0, 1))
    return gram.double().cpu()


def _gradient_parity(
    manual: Sequence[torch.Tensor | None],
    direct: Sequence[torch.Tensor | None],
) -> dict[str, Any]:
    if len(manual) != len(direct):
        raise ValueError("gradient parity tensor inventories differ")
    first = next(
        (
            tensor
            for pair in zip(manual, direct, strict=True)
            for tensor in pair
            if tensor is not None
        ),
        None,
    )
    if first is None:
        return {
            "none_mismatches": 0,
            "all_tensors_none": True,
            "max_abs": 0.0,
            "closure_defect_norm": 0.0,
            "manual_component_proxy_norm": 0.0,
            "direct_production_norm": 0.0,
            "relative_l2": None,
            "cosine": None,
            "manual_proxy_to_direct_norm_ratio": None,
            "relative_l2_gross_mismatch_tolerance": 0.05,
            "cosine_minimum": 0.995,
            "norm_ratio_bounds": [0.95, 1.05],
            "passed": False,
        }
    squared_delta = torch.zeros((), device=first.device)
    squared_direct = torch.zeros((), device=first.device)
    squared_manual = torch.zeros((), device=first.device)
    manual_direct_dot = torch.zeros((), device=first.device)
    max_abs = torch.zeros((), device=first.device)
    none_mismatches = 0
    for manual_tensor, direct_tensor in zip(manual, direct, strict=True):
        if (manual_tensor is None) != (direct_tensor is None):
            none_mismatches += 1
            continue
        if manual_tensor is None or direct_tensor is None:
            continue
        delta = manual_tensor.float() - direct_tensor.float()
        squared_delta.add_(delta.square().sum())
        squared_direct.add_(direct_tensor.float().square().sum())
        squared_manual.add_(manual_tensor.float().square().sum())
        manual_direct_dot.add_(
            (manual_tensor.float() * direct_tensor.float()).sum()
        )
        max_abs = torch.maximum(max_abs, delta.abs().amax())
    delta_norm = math.sqrt(float(squared_delta.item()))
    direct_norm = math.sqrt(float(squared_direct.item()))
    manual_norm = math.sqrt(float(squared_manual.item()))
    relative_l2 = delta_norm / max(direct_norm, 1.0e-30)
    cosine = (
        float(manual_direct_dot.item()) / (manual_norm * direct_norm)
        if manual_norm > 0.0 and direct_norm > 0.0
        else None
    )
    norm_ratio = manual_norm / direct_norm if direct_norm > 0.0 else None
    result = {
        "none_mismatches": none_mismatches,
        "all_tensors_none": False,
        "max_abs": float(max_abs.item()),
        "closure_defect_norm": delta_norm,
        "manual_component_proxy_norm": manual_norm,
        "direct_production_norm": direct_norm,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "manual_proxy_to_direct_norm_ratio": norm_ratio,
        "relative_l2_gross_mismatch_tolerance": 0.05,
        "cosine_minimum": 0.995,
        "norm_ratio_bounds": [0.95, 1.05],
        "interpretation": (
            "diagnostic non-additivity of separate BF16 component backwards; "
            "direct combined-objective backward is authoritative"
        ),
    }
    result["passed"] = (
        none_mismatches == 0
        and relative_l2 <= result["relative_l2_gross_mismatch_tolerance"]
        and cosine is not None
        and cosine >= result["cosine_minimum"]
        and norm_ratio is not None
        and result["norm_ratio_bounds"][0]
        <= norm_ratio
        <= result["norm_ratio_bounds"][1]
    )
    return result


def _objective_cancellation_evidence(
    *, component_norm_bound: float, direct_norm: float, closure_defect_norm: float
) -> dict[str, Any]:
    gap = float(component_norm_bound) - float(direct_norm)
    required = 3.0 * float(closure_defect_norm)
    supported = gap > 0.0 and gap >= required
    return {
        "component_proxy_norm_sum_bound": float(component_norm_bound),
        "direct_production_norm": float(direct_norm),
        "positive_cancellation_gap": max(gap, 0.0),
        "signed_gap": gap,
        "closure_defect_norm": float(closure_defect_norm),
        "required_gap_three_times_defect": required,
        "supported": supported,
        "verdict": (
            "supported_objective_cancellation"
            if supported
            else "inconclusive_at_three_times_closure_defect"
        ),
    }


def _closure_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    closures = [record["closure"] for record in records]
    if not closures:
        return {"microbatch_count": 0, "all_passed": True}

    def describe(values: Sequence[float]) -> dict[str, Any]:
        ordered = sorted(float(value) for value in values)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return {
            "values": [float(value) for value in values],
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p95_nearest_rank": ordered[p95_index],
            "max": ordered[-1],
        }

    return {
        "microbatch_count": len(closures),
        "all_passed": all(bool(row["passed"]) for row in closures),
        "none_mismatch_total": sum(int(row["none_mismatches"]) for row in closures),
        "relative_l2": describe([float(row["relative_l2"]) for row in closures]),
        "cosine": describe([float(row["cosine"]) for row in closures]),
        "manual_proxy_to_direct_norm_ratio": describe(
            [float(row["manual_proxy_to_direct_norm_ratio"]) for row in closures]
        ),
        "max_abs": describe([float(row["max_abs"]) for row in closures]),
    }


def _objective_batch_summary(
    gradients: Sequence[Sequence[torch.Tensor | None]],
    *,
    item_label: str,
    production_scale: float,
) -> dict[str, Any]:
    gram = _gradient_gram(gradients)
    norms = [math.sqrt(max(float(value), 0.0)) for value in gram.diag()]
    denominator = sum(norms)
    pairwise = [
        (
            float(gram[left, right]) / (norms[left] * norms[right])
            if norms[left] > 0.0 and norms[right] > 0.0
            else None
        )
        for left in range(len(gradients))
        for right in range(left + 1, len(gradients))
    ]
    summed_norm = math.sqrt(max(float(gram.sum()), 0.0))
    scaled_norms = [float(production_scale) * value for value in norms]
    scaled_sum_norm = float(production_scale) * summed_norm
    norm_mean = statistics.fmean(norms)
    return {
        "item_label": item_label,
        "item_count": len(gradients),
        "raw_item_norms": norms,
        "production_scaled_item_norms": scaled_norms,
        "raw_resultant_sum_norm": summed_norm,
        "production_scaled_resultant_sum_norm": scaled_sum_norm,
        "production_scaled_resultant_mean_norm": scaled_sum_norm
        / len(gradients),
        "resultant_over_sum_individual_norms": (
            summed_norm / denominator if denominator > 0.0 else None
        ),
        "individual_norm_cv": (
            statistics.pstdev(norms) / norm_mean if norm_mean > 0.0 else None
        ),
        "pairwise_cosines": pairwise,
        "none_tensor_counts": [
            sum(tensor is None for tensor in gradient) for gradient in gradients
        ],
    }


def _run_gradient_panel(
    *,
    model: torch.nn.Module,
    source: Any,
    cfg: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    print(
        "[v11-postmortem] stage=gradient-panel start batches=8 physical=B6x3",
        flush=True,
    )
    _names, parameters, groups = _gradient_groups(model)
    raw_model = operator_eval._raw_model(model)
    encoder = raw_model.four_trunk_complement_encoder_v11
    if encoder is None:
        raise RuntimeError("gradient panel lost V11 encoder")
    train_cfg = cfg["train"]
    struct = train_cfg["struct_loss"]
    structural_coef = float(train_cfg["structural_coef"])
    complement_coef = float(train_cfg["v11_complement_loss_weight"])
    behavioral_coef = float(train_cfg["behavioral_coef"])
    kl_beta = float(train_cfg["kl_beta"])
    use_latent_sampling = bool(cfg["model"]["big_vae"]["use_latent_sampling"])
    if (
        structural_coef != 1.0
        or complement_coef != 0.1
        or behavioral_coef != 0.0
        or kl_beta != 0.0
        or use_latent_sampling
    ):
        raise RuntimeError(
            "frozen V11 objective coefficients changed: "
            f"structural={structural_coef} complement={complement_coef} "
            f"behavioral={behavioral_coef} kl_beta={kl_beta} "
            f"use_latent_sampling={use_latent_sampling}"
        )
    group_cache: dict[tuple[Any, ...], Sequence[Any]] = {}
    rows: list[dict[str, Any]] = []
    closure_ledger: list[dict[str, Any]] = []
    panel_struct_sums: list[tuple[torch.Tensor | None, ...]] = []
    panel_complement_sums: list[tuple[torch.Tensor | None, ...]] = []
    panel_direct_sums: list[tuple[torch.Tensor | None, ...]] = []
    start_logical = REPLAY_STEPS * TILES_PER_STEP
    raw_model.train()
    try:
        for panel_step in range(GRADIENT_PANEL_STEPS):
            print(
                f"[v11-postmortem] stage=gradient-panel batch={panel_step + 1}/8",
                flush=True,
            )
            stored_struct: list[tuple[torch.Tensor | None, ...]] = []
            stored_complement: list[tuple[torch.Tensor | None, ...]] = []
            stored_direct: list[tuple[torch.Tensor | None, ...]] = []
            logical_by_micro: list[list[int]] = []
            identities: list[dict[str, Any]] = []
            per_micro_objective: list[dict[str, Any]] = []
            for micro in range(3):
                logical_start = start_logical + panel_step * 18 + micro * 6
                logical = list(range(logical_start, logical_start + 6))
                logical_by_micro.append(logical)
                batch = _materialize_logical_microbatch(source, logical, group_cache)
                W, x, x_mask, d_in_mask, d_out_mask, operators, tiles = [
                    value.to(device) for value in batch
                ]
                identities.append(
                    {
                        "logical_indices": logical,
                        "operator_indices": operators.cpu().tolist(),
                        "tile_indices": tiles.cpu().tolist(),
                    }
                )
                for trunk in encoder.trunks:
                    trunk.code_tensors_for_gradient.clear()
                    trunk.retain_code_gradient = True
                with worker._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    W_hat, mu, _logvar, pred_dirs = raw_model(
                        W,
                        x,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                    structural_loss, _details = WeightQuantileVAE.patch_structure_loss(
                        W,
                        W_hat,
                        patch_size=16,
                        gamma=float(struct["gamma"]),
                        lambda_dir=float(struct["lambda_dir"]),
                        lambda_scale=float(struct["lambda_scale"]),
                        lambda_rec=float(struct["lambda_rec"]),
                        lambda_rel=float(struct["lambda_rel"]),
                        huber_delta=float(struct["huber_delta"]),
                        pred_dirs=pred_dirs,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                    complement_loss = operator_eval.v11_complement_normalized_mse(
                        W,
                        W_hat,
                        mu,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                code_tensors = [trunk.last_code_for_gradient for trunk in encoder.trunks]
                if any(code is None for code in code_tensors):
                    raise RuntimeError("gradient panel missed serialized trunk code")
                targets = [*parameters, *code_tensors]
                production_loss = (
                    structural_coef * structural_loss
                    + complement_coef * complement_loss
                ) / 3.0
                grad_direct = torch.autograd.grad(
                    production_loss,
                    targets,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_struct = torch.autograd.grad(
                    structural_loss,
                    targets,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_complement = torch.autograd.grad(
                    complement_loss,
                    targets,
                    allow_unused=True,
                )
                if not all(
                    tensor is None or bool(torch.isfinite(tensor).all())
                    for tensor in (*grad_struct, *grad_complement, *grad_direct)
                ):
                    raise RuntimeError("gradient panel contains nonfinite gradients")
                manual_production = _gradient_add(
                    (grad_struct, grad_complement),
                    (structural_coef / 3.0, complement_coef / 3.0),
                )
                direct_parity = _gradient_parity(manual_production, grad_direct)
                closure_ledger.append(
                    {
                        "panel_step": panel_step,
                        "microbatch": micro,
                        "logical_indices": logical,
                        "structural_loss": float(structural_loss.detach().item()),
                        "complement_loss": float(complement_loss.detach().item()),
                        "closure": direct_parity,
                    }
                )
                _atomic_json(
                    output_root / "analysis" / "gradient_closure_partial.json",
                    {
                        "schema": SCHEMA,
                        "complete": False,
                        "distribution": _closure_distribution(closure_ledger),
                        "records": closure_ledger,
                    },
                )
                if not direct_parity["passed"]:
                    raise RuntimeError(
                        "separate BF16 component backwards have a gross mismatch "
                        f"from direct production backward: {direct_parity}"
                    )
                stored_struct.append(
                    tuple(
                        None if tensor is None else tensor.detach().to(torch.bfloat16)
                        for tensor in grad_struct
                    )
                )
                stored_complement.append(
                    tuple(
                        None if tensor is None else tensor.detach().to(torch.bfloat16)
                        for tensor in grad_complement
                    )
                )
                stored_direct.append(
                    tuple(
                        None if tensor is None else tensor.detach().to(torch.bfloat16)
                        for tensor in grad_direct
                    )
                )
                per_micro_objective.append(
                    {
                        "structural_loss": float(structural_loss.detach().item()),
                        "complement_loss": float(complement_loss.detach().item()),
                        "component_proxy_vs_direct_closure": direct_parity,
                    }
                )
                del (
                    grad_struct,
                    grad_complement,
                    grad_direct,
                    manual_production,
                    W_hat,
                    mu,
                    pred_dirs,
                )
            tiles_by_operator: dict[int, list[int]] = {}
            for identity in identities:
                for operator, tile in zip(
                    identity["operator_indices"],
                    identity["tile_indices"],
                    strict=True,
                ):
                    tiles_by_operator.setdefault(int(operator), []).append(int(tile))
            if len(tiles_by_operator) != 2 or any(
                len(tiles) != 9 or set(tiles) != set(range(9))
                for tiles in tiles_by_operator.values()
            ):
                raise RuntimeError(
                    "gradient B18 must contain exactly two operators with one copy "
                    f"of every tile 0..8: {tiles_by_operator}"
                )
            extended_groups = dict(groups)
            for trunk in range(4):
                extended_groups[f"serialized_code_{trunk:02d}"] = [
                    len(parameters) + trunk
                ]
            panel_struct_sums.append(
                tuple(
                    None if tensor is None else tensor.to(torch.bfloat16)
                    for tensor in _gradient_add(
                        stored_struct, (1.0, 1.0, 1.0)
                    )
                )
            )
            panel_complement_sums.append(
                tuple(
                    None if tensor is None else tensor.to(torch.bfloat16)
                    for tensor in _gradient_add(
                        stored_complement, (1.0, 1.0, 1.0)
                    )
                )
            )
            panel_direct_sums.append(
                tuple(
                    None if tensor is None else tensor.to(torch.bfloat16)
                    for tensor in _gradient_add(
                        stored_direct, (1.0, 1.0, 1.0)
                    )
                )
            )
            group_rows: dict[str, Any] = {}
            scale_ratios: dict[str, float | None] = {}
            for group_name, indices in extended_groups.items():
                struct_group = [
                    _select_gradient(gradient, indices) for gradient in stored_struct
                ]
                comp_group = [
                    _select_gradient(gradient, indices)
                    for gradient in stored_complement
                ]
                direct_group = [
                    _select_gradient(gradient, indices)
                    for gradient in stored_direct
                ]
                struct_sum = _gradient_add(struct_group, (1.0, 1.0, 1.0))
                comp_sum = _gradient_add(comp_group, (1.0, 1.0, 1.0))
                direct_sum = _gradient_add(direct_group, (1.0, 1.0, 1.0))
                manual_proxy_sum = _gradient_add(
                    (struct_sum, comp_sum), (1.0 / 3.0, 0.1 / 3.0)
                )
                group_closure = _gradient_parity(manual_proxy_sum, direct_sum)
                struct_norm = _gradient_norm(struct_sum)
                comp_norm = _gradient_norm(comp_sum)
                direct_norm = _gradient_norm(direct_sum)
                component_norm_bound = (struct_norm + 0.1 * comp_norm) / 3.0
                scale_ratios[group_name] = (
                    struct_norm / comp_norm if comp_norm > 0.0 else None
                )
                group_rows[group_name] = {
                    "raw_structural": _objective_batch_summary(
                        struct_group,
                        item_label="B6_microbatch_raw_gradient",
                        production_scale=1.0 / 3.0,
                    ),
                    "raw_complement": _objective_batch_summary(
                        comp_group,
                        item_label="B6_microbatch_raw_gradient",
                        production_scale=0.1 / 3.0,
                    ),
                    "actual_total_direct_production_backward": _objective_batch_summary(
                        direct_group,
                        item_label="B6_direct_production_gradient",
                        production_scale=1.0,
                    ),
                    "b18_structural_complement_cosine": _cosine(
                        struct_sum, comp_sum
                    ),
                    "b18_raw_structural_norm": struct_norm,
                    "b18_raw_complement_norm": comp_norm,
                    "b18_production_structural_proxy_norm": struct_norm / 3.0,
                    "b18_production_weighted_complement_proxy_norm": 0.1
                    * comp_norm
                    / 3.0,
                    "b18_production_actual_total_norm": direct_norm,
                    "component_closure_direct_minus_manual": group_closure,
                    "objective_interference_ratio": (
                        direct_norm / component_norm_bound
                        if component_norm_bound > 0.0
                        else None
                    ),
                    "objective_cancellation_evidence": _objective_cancellation_evidence(
                        component_norm_bound=component_norm_bound,
                        direct_norm=direct_norm,
                        closure_defect_norm=float(
                            group_closure["closure_defect_norm"]
                        ),
                    ),
                    "raw_gradient_scale_ratio_struct_over_comp": scale_ratios[
                        group_name
                    ],
                }
            row = {
                "panel_step": panel_step,
                "source_optimizer_step": REPLAY_STEPS + panel_step + 1,
                "logical_start": logical_by_micro[0][0],
                "logical_end_exclusive": logical_by_micro[-1][-1] + 1,
                "microbatches": identities,
                "b18_tiles_by_operator": {
                    str(operator): tiles
                    for operator, tiles in sorted(tiles_by_operator.items())
                },
                "losses": per_micro_objective,
                "groups": group_rows,
            }
            rows.append(row)
            _atomic_json(
                output_root / "analysis" / "gradient_panel_partial.json",
                {"schema": SCHEMA, "complete": False, "batches": rows},
            )
            del stored_struct, stored_complement, stored_direct
            torch.cuda.empty_cache()
    finally:
        for trunk in encoder.trunks:
            trunk.retain_code_gradient = False
            trunk.last_code_for_gradient = None
            trunk.code_tensors_for_gradient.clear()
        raw_model.eval()
    global_ratios = [
        float(row["groups"]["all_trainable"]["raw_gradient_scale_ratio_struct_over_comp"])
        for row in rows
    ]
    cross_b18: dict[str, Any] = {}
    extended_groups = dict(groups)
    for trunk in range(4):
        extended_groups[f"serialized_code_{trunk:02d}"] = [len(parameters) + trunk]
    for group_name, indices in extended_groups.items():
        struct_batches = [
            _select_gradient(gradient, indices) for gradient in panel_struct_sums
        ]
        comp_batches = [
            _select_gradient(gradient, indices)
            for gradient in panel_complement_sums
        ]
        direct_batches = [
            _select_gradient(gradient, indices) for gradient in panel_direct_sums
        ]
        struct_panel_sum = _gradient_add(
            struct_batches, [1.0] * len(struct_batches)
        )
        comp_panel_sum = _gradient_add(comp_batches, [1.0] * len(comp_batches))
        direct_panel_sum = _gradient_add(
            direct_batches, [1.0] * len(direct_batches)
        )
        manual_proxy_panel_sum = _gradient_add(
            (struct_panel_sum, comp_panel_sum), (1.0 / 3.0, 0.1 / 3.0)
        )
        panel_closure = _gradient_parity(
            manual_proxy_panel_sum, direct_panel_sum
        )
        struct_panel_norm = _gradient_norm(struct_panel_sum)
        comp_panel_norm = _gradient_norm(comp_panel_sum)
        direct_panel_norm = _gradient_norm(direct_panel_sum)
        panel_component_norm_bound = (
            struct_panel_norm + 0.1 * comp_panel_norm
        ) / 3.0
        cross_b18[group_name] = {
            "raw_structural": _objective_batch_summary(
                struct_batches,
                item_label="B18_raw_sum_gradient",
                production_scale=1.0 / 3.0,
            ),
            "raw_complement": _objective_batch_summary(
                comp_batches,
                item_label="B18_raw_sum_gradient",
                production_scale=0.1 / 3.0,
            ),
            "actual_total_direct_production_backward": _objective_batch_summary(
                direct_batches,
                item_label="B18_direct_production_gradient",
                production_scale=1.0,
            ),
            "panel_structural_complement_cosine": _cosine(
                struct_panel_sum,
                comp_panel_sum,
            ),
            "panel_production_structural_resultant_norm": struct_panel_norm / 3.0,
            "panel_production_weighted_complement_resultant_norm": 0.1
            * comp_panel_norm
            / 3.0,
            "panel_production_actual_resultant_norm": direct_panel_norm,
            "panel_component_closure_direct_minus_manual": panel_closure,
            "panel_objective_interference_ratio": (
                direct_panel_norm / panel_component_norm_bound
                if panel_component_norm_bound > 0.0
                else None
            ),
            "panel_objective_cancellation_evidence": _objective_cancellation_evidence(
                component_norm_bound=panel_component_norm_bound,
                direct_norm=direct_panel_norm,
                closure_defect_norm=float(panel_closure["closure_defect_norm"]),
            ),
        }
    result = {
        "schema": SCHEMA,
        "snapshot_step": REPLAY_STEPS,
        "logical_range": [start_logical, start_logical + GRADIENT_PANEL_STEPS * 18],
        "physical_batch": 6,
        "grad_accum_steps": 3,
        "storage_dtype": (
            "separate component and direct production gradients stored bfloat16; "
            "FP32 accumulation for all dot products"
        ),
        "actual_objective": (
            "authoritative direct backward of "
            "(structural + 0.1 * complement) / grad_accum_steps"
        ),
        "actual_quantity": (
            "objective/backward gradient at the fixed step256 weights; "
            "not an optimizer update"
        ),
        "component_combination_policy": (
            "never synthesize actual gradients from separate BF16 components; "
            "manual-vs-direct is diagnostic only"
        ),
        "global_raw_scale_ratio_median": statistics.median(global_ratios),
        "global_raw_scale_ratio_values": global_ratios,
        "global_component_closure_gate": {
            **_closure_distribution(closure_ledger),
            "precommitted_thresholds": {
                "relative_l2_max": 0.05,
                "cosine_min": 0.995,
                "norm_ratio_bounds": [0.95, 1.05],
                "none_mismatches": 0,
            },
        },
        "cross_b18_groups": cross_b18,
        "batches": rows,
    }
    _atomic_json(output_root / "analysis" / "gradient_panel.json", result)
    _atomic_json(
        output_root / "analysis" / "gradient_closure.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "distribution": _closure_distribution(closure_ledger),
            "records": closure_ledger,
        },
    )
    (output_root / "analysis" / "gradient_closure_partial.json").unlink(
        missing_ok=True
    )
    print("[v11-postmortem] stage=gradient-panel complete", flush=True)
    return result


def _scientific_config_binding(
    *,
    cfg: Mapping[str, Any],
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    failed_cfg = json.loads(
        (FAILED_ROOT / "resolved_run_config.json").read_text(encoding="utf-8")
    )
    failed_manifest = json.loads(
        (FAILED_ROOT / "operator_set_selection.json").read_text(encoding="utf-8")
    )
    if _canonical_json(cfg) != _canonical_json(failed_cfg):
        differing = sorted(
            key
            for key in set(cfg) | set(failed_cfg)
            if cfg.get(key) != failed_cfg.get(key)
        )
        raise RuntimeError(
            f"supplied V11 config does not reconstruct failed resolved config: {differing}"
        )
    if _canonical_json(manifest) != _canonical_json(failed_manifest):
        raise RuntimeError("supplied V11 selection does not reconstruct failed selection")
    overfit = cfg["train"]["operator_bank"]["operator_set_overfit"]
    selected = overfit["selected_operators"]
    supplied_contract = {
        "output_root": str(Path(str(spec["output_root"])).expanduser().resolve()),
        "run_id": str(spec["run_id"]),
        "seed": int(spec["seed"]),
        "layer_key": str(spec["layer_key"]),
        "operator_count": int(spec["operator_count"]),
        "steps": int(spec["steps"]),
        "physical_batch": int(spec["physical_batch"]),
        "grad_accum_steps": int(spec["grad_accum_steps"]),
        "eval_every_steps": int(spec["eval_every_steps"]),
        "eval_batch_size": int(spec["eval_batch_size"]),
        "comet_enabled": bool(spec["comet_enabled"]),
    }
    resolved_contract = {
        "output_root": str(Path(cfg["training_artifacts"]["root_dir"]).resolve()),
        "run_id": str(cfg["run_config"]["name"]),
        "seed": int(cfg["data"]["seed"]),
        "layer_key": sorted({str(row["layer_key"]) for row in selected}),
        "operator_count": len(selected),
        "steps": int(cfg["train"]["max_steps"]),
        "stop_after_step": int(cfg["train"]["stop_after_step"]),
        "physical_batch": int(cfg["train"]["slice_batch_size"]),
        "grad_accum_steps": int(cfg["train"]["grad_accum_steps"]),
        "eval_every_steps": int(overfit["eval_every_steps"]),
        "eval_batch_size": int(overfit["eval_batch_size"]),
        "comet_enabled": bool(cfg["train"]["telemetry"]["comet"]["enabled"]),
    }
    expected_resolved = {
        **supplied_contract,
        "layer_key": [supplied_contract["layer_key"]],
        "stop_after_step": supplied_contract["steps"],
    }
    if _canonical_json(resolved_contract) != _canonical_json(expected_resolved):
        raise RuntimeError(
            "supplied YAML scientific contract differs from the failed resolved "
            f"config: supplied={expected_resolved} resolved={resolved_contract}"
        )
    if _canonical_json(selected) != _canonical_json(manifest["operators"]):
        raise RuntimeError("resolved selected operators differ from bound manifest")
    scientific = {
        "architecture": cfg["model"]["big_vae"]["architecture_version"],
        "seed": cfg["data"]["seed"],
        "operator_set": cfg["train"]["operator_bank"]["operator_set_overfit"],
        "physical_batch": cfg["train"]["slice_batch_size"],
        "grad_accum_steps": cfg["train"]["grad_accum_steps"],
        "structural_coef": cfg["train"]["structural_coef"],
        "behavioral_coef": cfg["train"]["behavioral_coef"],
        "struct_loss": cfg["train"]["struct_loss"],
        "v11_complement_loss_weight": cfg["train"][
            "v11_complement_loss_weight"
        ],
        "kl_beta": cfg["train"]["kl_beta"],
        "use_latent_sampling": cfg["model"]["big_vae"]["use_latent_sampling"],
        "optimizer": {
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "scheduler_name": cfg["train"]["scheduler_name"],
        },
    }
    return {
        "config_exact_equal": True,
        "selection_exact_equal": True,
        "supplied_yaml_scientific_contract_exact": True,
        "supplied_contract": supplied_contract,
        "scientific_contract_sha256": hashlib.sha256(
            _canonical_json(scientific)
        ).hexdigest(),
        "resolved_config_sha256": FAILED_ARTIFACT_HASHES["resolved_config"],
        "selection_sha256": FAILED_ARTIFACT_HASHES["selection"],
    }


def main() -> None:
    args = _parse_args()
    workspace = Path(__file__).resolve().parents[2]
    config_path = args.config.expanduser().resolve()
    spec = _load_spec(config_path)
    official_spec = _load_spec((workspace / DEFAULT_SPEC).resolve())
    if _canonical_json(spec) != _canonical_json(official_spec):
        raise RuntimeError("supplied postmortem config is not the frozen V11 spec")
    cfg = json.loads(
        (FAILED_ROOT / "resolved_run_config.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (FAILED_ROOT / "operator_set_selection.json").read_text(encoding="utf-8")
    )
    source_binding = _source_binding(workspace)
    source_binding[str(Path(__file__).resolve().relative_to(workspace))] = {
        "sha256": sha256_file(Path(__file__).resolve())
    }
    failed_binding = _validate_failed_run_binding()
    config_binding = _scientific_config_binding(
        cfg=cfg, manifest=manifest, spec=spec
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _default_output_root()
    )
    failed_root_resolved = FAILED_ROOT.resolve()
    if output_root == failed_root_resolved or output_root.is_relative_to(
        failed_root_resolved
    ):
        raise ValueError(
            "postmortem output root must not equal or descend from the frozen "
            f"failed root: {output_root}"
        )
    startup = {
        "schema": SCHEMA,
        "mode": "dry_run" if args.dry_run else "run",
        "device": cfg["train"]["device"],
        "dtype": cfg["train"].get("amp_dtype", "bf16"),
        "seed": cfg["data"]["seed"],
        "output_root": str(output_root),
        "failed_root": str(FAILED_ROOT),
        "replay_steps": REPLAY_STEPS,
        "committed_cursor": REPLAY_STEPS * TILES_PER_STEP,
        "gradient_panel": "8 consecutive B18 batches (B6x3)",
        "ridge": "exact64 grouped operator split 48/16",
        "forks_omitted": True,
        "checkpoint_policy": "ephemeral /dev/shm resume only; delete in finally",
        "config_binding": config_binding,
    }
    print("[v11-postmortem] stage=preflight", flush=True)
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    if output_root.exists():
        raise FileExistsError(f"postmortem output root already exists: {output_root}")
    if args.dry_run:
        print(
            "[v11-postmortem] stage=dry-run-complete no_files_written=true",
            flush=True,
        )
        return

    output_root.mkdir(parents=True, exist_ok=False)
    ephemeral_root = Path(
        tempfile.mkdtemp(prefix="v11_postmortem_", dir="/dev/shm")
    )
    partial_path = output_root / "report.partial.json"
    final_path = output_root / "report.json"
    report: dict[str, Any] = {
        **startup,
        "complete": False,
        "source_binding": source_binding,
        "failed_run_binding": failed_binding,
        "started_unix": time.time(),
        "stages": {},
    }
    _atomic_json(partial_path, report)
    try:
        replay_cfg = _rebase_worker_config(
            cfg, output_root=output_root, ephemeral_root=ephemeral_root
        )
        _atomic_json(output_root / "replay" / "resolved_replay_config.json", replay_cfg)
        resume_path, replay_result = _run_replay(
            cfg=replay_cfg,
            output_root=output_root,
            ephemeral_root=ephemeral_root,
        )
        report["stages"]["replay"] = replay_result
        _atomic_json(partial_path, report)

        replay_rows = [
            json.loads(line)
            for line in (
                output_root / "replay" / "operator_set_metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [int(float(row["step"])) for row in replay_rows] != [0, 256]:
            raise RuntimeError("bounded replay metric steps are not exactly [0,256]")
        (
            model,
            payload,
            device,
            amp_enabled,
            amp_dtype,
            runtime_restore,
        ) = _load_snapshot_model(cfg=replay_cfg, resume_path=resume_path)
        parameter_versions = {
            name: int(parameter._version)
            for name, parameter in model.named_parameters()
        }
        state_digest_before = _model_state_digest(model)
        report["stages"]["snapshot"] = {
            "step": int(payload["step"]),
            "committed_cursor": int(
                payload["rng_state"]["data_stream"]["logical_sample_index"]
            ),
            "model_tensors": len(payload["model_state"]),
            "optimizer_groups": len(payload["optimizer_state"]["param_groups"]),
            "runtime_restore": runtime_restore,
            "model_state_digest_before_analysis": state_digest_before,
            "state_persisted": False,
        }
        _atomic_json(partial_path, report)

        with _build_source_context(replay_cfg) as (dataset, _sampler):
            source = dataset.source
            evaluator, reload_row = _reload_parity_eval(
                model=model,
                source=source,
                output_root=output_root,
                archived_row=replay_rows[1],
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            report["stages"]["reload_parity"] = {
                name: reload_row[name] for name in ARCHIVED_STEP256
            }
            _atomic_json(partial_path, report)

            worker._restore_training_rng_state(
                payload["rng_state"], active_cuda_device=device
            )
            report["stages"]["gradient_rng_reset"] = {
                "source": "post-update/post-eval step256 snapshot",
                "reason": (
                    "discard any setup/reload-evaluation RNG consumption before "
                    "the exact logical 4608 gradient panel"
                ),
            }
            gradient = _run_gradient_panel(
                model=model,
                source=source,
                cfg=replay_cfg,
                output_root=output_root,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            report["stages"]["gradient_panel"] = {
                "logical_range": gradient["logical_range"],
                "global_raw_scale_ratio_median": gradient[
                    "global_raw_scale_ratio_median"
                ],
                "artifact": str(output_root / "analysis" / "gradient_panel.json"),
            }
            _atomic_json(partial_path, report)

            table = _capture_ridge_table(
                model=model,
                evaluator=evaluator,
                source=source,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            ridge = _run_ridge(table=table, output_root=output_root, device=device)
            report["stages"]["ridge"] = {
                "artifact": str(output_root / "analysis" / "ridge_report.json"),
                "features": {
                    row["feature"]: row["test_normalized_mse"]
                    for row in ridge["features"]
                },
            }
            _atomic_json(partial_path, report)

        changed_versions = [
            name
            for name, parameter in model.named_parameters()
            if int(parameter._version) != parameter_versions[name]
        ]
        if changed_versions:
            raise RuntimeError(
                f"zero-update analysis mutated model parameters: {changed_versions[:8]}"
            )
        state_digest_after = _model_state_digest(model)
        if state_digest_after != state_digest_before:
            raise RuntimeError(
                "zero-update analysis changed parameter/buffer content: "
                f"before={state_digest_before} after={state_digest_after}"
            )
        report["stages"]["no_update_proof"] = {
            "parameter_count": len(parameter_versions),
            "changed_parameter_versions": 0,
            "optimizer_restore_only": True,
            "optimizer_step_calls_after_snapshot": 0,
            "backward_api": "torch.autograd.grad only",
            "model_state_digest_before": state_digest_before,
            "model_state_digest_after": state_digest_after,
            "all_parameters_and_buffers_identical": True,
        }
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["failed_unix"] = time.time()
        _atomic_json(partial_path, report)
        raise
    finally:
        shutil.rmtree(ephemeral_root, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ephemeral_root.exists():
        report["failure"] = {
            "type": "EphemeralCleanupError",
            "message": f"ephemeral snapshot root still exists: {ephemeral_root}",
        }
        report["failed_unix"] = time.time()
        _atomic_json(partial_path, report)
        raise RuntimeError(report["failure"]["message"])
    report["complete"] = True
    report["completed_unix"] = time.time()
    report["ephemeral_snapshot_deleted"] = True
    _atomic_json(final_path, report)
    _atomic_json(
        output_root / "COMPLETE.json",
        {
            "schema": SCHEMA,
            "report": str(final_path),
            "report_sha256": sha256_file(final_path),
            "complete": True,
        },
    )
    partial_path.unlink(missing_ok=True)
    print(
        "[v11-postmortem] stage=complete "
        f"report={final_path} gradient={output_root / 'analysis' / 'gradient_panel.json'} "
        f"ridge={output_root / 'analysis' / 'ridge_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
