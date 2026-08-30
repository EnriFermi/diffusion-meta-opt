from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .objective import conditional_flow_matching_loss, controlled_objective_mask
from .solvers import integrate_flow


MATCHED_FLOW_EFFECTIVE_BATCH_SIZE = 256


@dataclass(slots=True)
class FlowTrainConfig:
    experiment_label: str
    path_kind: str
    output_dir: str
    steps: int = 100_000
    effective_batch_size: int = 256
    microbatch_size: int = 32
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    validation_interval: int = 2_000
    checkpoint_interval: int = 2_000
    log_interval: int = 50
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    resume: bool = True
    cache_mode: str = "grouped_latent_records"
    validation_solver: str = "heun"
    validation_nfe_steps: int = 16


class EMA:
    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            shadow = self.shadow[name]
            if value.is_floating_point():
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if str(key).lower() in {"auth_token", "access_token", "hf_token", "api_key", "password", "secret"}
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _repair_history_for_resume(path: Path, *, start_step: int, has_checkpoint: bool) -> None:
    """Atomically discard uncheckpointed tail rows and reject ambiguous history."""

    if not path.exists():
        if start_step > 0:
            raise ValueError("resume checkpoint exists but metrics history is missing")
        return
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed flow history line {line_number}") from exc
        if not isinstance(row, dict) or row.get("stage") not in {"train", "validation"}:
            raise ValueError(f"invalid flow history row at line {line_number}")
        step = row.get("step")
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f"invalid flow history step at line {line_number}")
        identity = (str(row["stage"]), step)
        if identity in seen:
            raise ValueError(f"duplicate flow history row {identity}")
        seen.add(identity)
        rows.append(row)
    if not has_checkpoint and rows:
        raise ValueError("metrics history exists without a resumable checkpoint")
    if start_step > 0 and ("train", start_step) not in seen:
        raise ValueError("metrics history does not contain the resumed checkpoint step")
    retained = [row for row in rows if int(row["step"]) <= start_step]
    if len(retained) != len(rows):
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(path)
@contextmanager
def _ema_weights(model: nn.Module, ema: EMA):
    current = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema.shadow, strict=True)
    try:
        yield
    finally:
        model.load_state_dict(current, strict=True)


@torch.no_grad()
def validate_flow(
    model: nn.Module,
    loader: Any,
    *,
    path_kind: str,
    device: torch.device,
    solver: str,
    nfe_steps: int,
    seed: int = 0,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total, total_endpoint_sq, total_prediction_sq, total_target_sq = 0.0, 0.0, 0.0, 0.0
    total_cosine, total_finite, total_nonbody_exact, count = 0.0, 0.0, 0.0, 0
    for batch_index, batch in enumerate(loader):
        batch = _move_batch(batch, device)
        codecs = {str(codec) for codec in batch.get("codec", [])}
        is_weightclip = codecs == {"weightclip"}
        if codecs and codecs not in ({"ours"}, {"weightclip"}):
            raise ValueError(f"validation batch has unsupported/mixed codecs: {sorted(codecs)}")
        generator = torch.Generator(device=device).manual_seed(int(seed) + batch_index)
        output = conditional_flow_matching_loss(model, batch, path_kind=path_kind, generator=generator)
        n = int(batch["z_task"].shape[0])
        total += float(output.velocity_mse.item()) * n
        if path_kind == "paired_anchor":
            base = batch["z_enc"]
        else:
            base = torch.randn(
                batch["z_task"].shape,
                dtype=batch["z_task"].dtype,
                device=device,
                generator=generator,
            )

        def velocity(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            attention_mask = controlled_objective_mask(batch)
            value = model(
                z,
                t,
                dataset_embedding=batch["dataset_embedding"],
                architecture_features=batch["architecture_features"],
                token_mask=attention_mask,
            )
            if is_weightclip:
                body = batch["architecture_features"][..., 1].to(device=device, dtype=value.dtype).unsqueeze(-1)
                value = value * body
            return value

        endpoint = integrate_flow(velocity, base, method=solver, steps=nfe_steps).endpoint.float()
        if path_kind == "paired_anchor":
            body = batch["architecture_features"][..., 1].to(device=device, dtype=torch.bool).unsqueeze(-1)
            endpoint = torch.where(body, endpoint, base.float())
            nonbody = ~body.expand_as(endpoint)
            exact = 1.0 if not nonbody.any() else float(torch.equal(endpoint[nonbody], base.float()[nonbody]))
        else:
            exact = 1.0
        target = batch["z_task"].float()
        mask = controlled_objective_mask(batch)
        if mask is None:
            mask_f = torch.ones_like(target[..., 0])
        else:
            mask_f = mask.to(device=device, dtype=target.dtype)
        expanded = mask_f.unsqueeze(-1)
        denom = (mask_f.sum() * target.shape[-1]).clamp_min(1.0)
        endpoint_sq = ((endpoint - target).square() * expanded).sum() / denom
        prediction_sq = (endpoint.square() * expanded).sum() / denom
        target_sq = (target.square() * expanded).sum() / denom
        flat_endpoint = (endpoint * expanded).reshape(n, -1)
        flat_target = (target * expanded).reshape(n, -1)
        cosine = torch.nn.functional.cosine_similarity(flat_endpoint, flat_target, dim=-1).mean()
        finite = torch.isfinite(endpoint).reshape(n, -1).all(dim=1).float().mean()
        total_endpoint_sq += float(endpoint_sq.item()) * n
        total_prediction_sq += float(prediction_sq.item()) * n
        total_target_sq += float(target_sq.item()) * n
        total_cosine += float(cosine.item()) * n
        total_finite += float(finite.item()) * n
        total_nonbody_exact += exact * n
        count += n
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if count == 0:
        raise ValueError("validation loader is empty")
    endpoint_mse = total_endpoint_sq / count
    target_rms = (total_target_sq / count) ** 0.5
    prediction_rms = (total_prediction_sq / count) ** 0.5
    return {
        "velocity_mse": total / count,
        "ema_endpoint_mse": endpoint_mse,
        "ema_endpoint_rmse": endpoint_mse**0.5,
        "ema_endpoint_cosine": total_cosine / count,
        "ema_endpoint_norm_ratio": prediction_rms / max(target_rms, 1e-12),
        "ema_endpoint_finite_fraction": total_finite / count,
        "paired_nonbody_bitwise_fraction": total_nonbody_exact / count,
        "solver_nfe": float(nfe_steps * (2 if solver == "heun" else 1)),
        "examples": float(count),
    }


def _resume_contract(model: nn.Module, config: FlowTrainConfig, provenance: Mapping[str, Any]) -> str:
    payload = {
        "model_config": asdict(model.config) if hasattr(model, "config") else {},
        "train_config": asdict(config),
        "provenance": _redact(dict(provenance)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def train_flow(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    config: FlowTrainConfig,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if config.path_kind not in {"gaussian", "paired_anchor"}:
        raise ValueError("path_kind must be gaussian or paired_anchor")
    assert_matched_sample_exposure(config)
    if config.checkpoint_interval % config.validation_interval != 0:
        raise ValueError("every flow checkpoint must coincide with an EMA endpoint validation")
    codec_fingerprint = provenance.get("codec_fingerprint")
    if not isinstance(codec_fingerprint, str) or not codec_fingerprint.startswith("sha256:") or len(codec_fingerprint) != 71:
        raise ValueError("flow training requires one canonical grouped-record codec_fingerprint")
    _set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}.get(config.dtype)
    if dtype is None:
        raise ValueError(f"unsupported dtype {config.dtype!r}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resolved = _redact(
        {
            "train": asdict(config),
            "model": asdict(model.config) if hasattr(model, "config") else {},
            "provenance": dict(provenance),
        }
    )
    config_path = output_dir / "resolved_config.json"
    resolved_bytes = (json.dumps(resolved, indent=2, sort_keys=True) + "\n").encode()
    if config_path.exists():
        if config_path.read_bytes() != resolved_bytes:
            attempted_hash = hashlib.sha256(resolved_bytes).hexdigest()[:16]
            attempted_path = output_dir / f"attempted_config_{attempted_hash}.json"
            if not attempted_path.exists():
                attempted_path.write_bytes(resolved_bytes)
            raise ValueError(
                f"immutable resolved flow config mismatch; original preserved at {config_path}, "
                f"attempt recorded at {attempted_path}"
            )
    else:
        temporary_config = config_path.with_suffix(".json.tmp")
        temporary_config.write_bytes(resolved_bytes)
        temporary_config.replace(config_path)
    print(
        json.dumps(
            {
                "stage": "flow_start",
                "experiment": config.experiment_label,
                "path_kind": config.path_kind,
                "device": str(device),
                "dtype": str(dtype),
                "seed": config.seed,
                "cache_mode": config.cache_mode,
                "output_dir": str(output_dir.resolve()),
                "config_path": str(config_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resume_contract_sha256 = _resume_contract(model, config, provenance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.min_learning_rate)
    ema = EMA(model, config.ema_decay)
    checkpoint_path = output_dir / "latest.pt"
    start_step = 0
    if config.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(state.get("schema_version", -1)) != 3:
            raise ValueError("resume checkpoint predates immutable flow resume contract v3")
        if state.get("provenance", {}).get("codec_fingerprint") != codec_fingerprint:
            raise ValueError("resume checkpoint codec fingerprint differs from current grouped records")
        if state.get("resume_contract_sha256") != resume_contract_sha256:
            raise ValueError("resume checkpoint config/data/normalizer contract differs from current run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema"])
        start_step = int(state["step"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(json.dumps({"stage": "cache_hit_resume", "path": str(checkpoint_path.resolve()), "step": start_step}), flush=True)
    elif config.resume:
        print(json.dumps({"stage": "cache_miss_resume", "path": str(checkpoint_path.resolve())}), flush=True)
    history_path = output_dir / "metrics.jsonl"
    _repair_history_for_resume(history_path, start_step=start_step, has_checkpoint=checkpoint_path.exists())
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_start_step"):
        start_microstep = start_step * config.grad_accum_steps
        batch_sampler.set_start_step(start_microstep)
        print(
            json.dumps(
                {
                    "stage": "batch_stream_resume",
                    "start_step": start_step,
                    "start_microstep": start_microstep,
                    "sampler": type(batch_sampler).__name__,
                }
            ),
            flush=True,
        )
    elif start_step > 0:
        raise RuntimeError("exact resume requires a step-addressable train batch sampler")
    iterator = iter(train_loader)
    started = time.monotonic()
    model.train()
    last_step_finished = time.monotonic()
    last_validation: dict[str, float] | None = None
    for step in range(start_step + 1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_velocity = 0.0
        accumulated_path_rms = 0.0
        codec_valid_token_updates = 0
        controlled_body_token_updates = 0
        dense_attention_pair_updates = 0
        for microstep in range(config.grad_accum_steps):
            batch, iterator = _next_batch(iterator, train_loader)
            batch = _move_batch(batch, device)
            use_autocast = device.type == "cuda" and dtype == torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                output = conditional_flow_matching_loss(model, batch, path_kind=config.path_kind)
                scaled_loss = output.loss / config.grad_accum_steps
            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"non-finite flow loss at step {step}, microstep {microstep}")
            scaled_loss.backward()
            accumulated_loss += float(output.loss.item()) / config.grad_accum_steps
            accumulated_velocity += float(output.velocity_mse.item()) / config.grad_accum_steps
            accumulated_path_rms += float(output.endpoint_distance.item()) / config.grad_accum_steps
            codec_valid_token_updates += int(batch["token_mask"].sum().item())
            controlled_body_token_updates += int(controlled_objective_mask(batch).sum().item())
            dense_attention_pair_updates += int(
                batch["z_task"].shape[0] * batch["z_task"].shape[1] ** 2 * model.config.depth
            )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item())
        optimizer.step()
        scheduler.step()
        ema.update(model)
        step_finished = time.monotonic()
        step_seconds = max(step_finished - last_step_finished, 1e-12)
        last_step_finished = step_finished
        row = {
            "stage": "train",
            "experiment": config.experiment_label,
            "step": step,
            "steps": config.steps,
            "loss": accumulated_loss,
            "velocity_mse": accumulated_velocity,
            "path_endpoint_rms": accumulated_path_rms,
            "grad_norm": grad_norm,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.monotonic() - started,
            "effective_batch_size": config.effective_batch_size,
            "microbatch_size": config.microbatch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "examples_per_sec": config.effective_batch_size / step_seconds,
            "codec_valid_token_updates": codec_valid_token_updates,
            "controlled_body_token_updates": controlled_body_token_updates,
            "controlled_body_tokens_per_sec": controlled_body_token_updates / step_seconds,
            "dense_attention_pair_updates": dense_attention_pair_updates,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        if step % config.log_interval == 0 or step == start_step + 1:
            print(json.dumps(row, sort_keys=True), flush=True)
        with history_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if step % config.validation_interval == 0 or step == config.steps:
            with _ema_weights(model, ema):
                validation = validate_flow(
                    model,
                    validation_loader,
                    path_kind=config.path_kind,
                    device=device,
                    solver=config.validation_solver,
                    nfe_steps=config.validation_nfe_steps,
                    seed=config.seed,
                )
            last_validation = validation
            val_row = {"stage": "validation", "experiment": config.experiment_label, "step": step, **validation}
            print(json.dumps(val_row, sort_keys=True), flush=True)
            with history_path.open("a") as handle:
                handle.write(json.dumps(val_row, sort_keys=True) + "\n")
            model.train()
        if step % config.checkpoint_interval == 0 or step == config.steps:
            if last_validation is None:
                raise RuntimeError("refusing to checkpoint a flow without EMA endpoint validation")
            checkpoint = {
                "schema_version": 3,
                "step": step,
                "model": model.state_dict(),
                "model_config": asdict(model.config) if hasattr(model, "config") else {},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "train_config": asdict(config),
                "provenance": _redact(dict(provenance)),
                "resume_contract_sha256": resume_contract_sha256,
                "validation_metrics": last_validation,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(checkpoint_path)
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            print(json.dumps({"stage": "checkpoint", "step": step, "path": str(checkpoint_path.resolve()), "sha256": digest}), flush=True)
    summary = {
        "experiment": config.experiment_label,
        "step": config.steps,
        "checkpoint": str(checkpoint_path.resolve()),
        "metrics": str(history_path.resolve()),
        "elapsed_sec": time.monotonic() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"stage": "flow_complete", **summary, "summary": str(summary_path.resolve())}, sort_keys=True), flush=True)
    return summary


def assert_matched_flow_budgets(models: Mapping[str, nn.Module]) -> dict[str, dict[str, int]]:
    ledgers = {name: model.parameter_ledger() for name, model in models.items()}
    core_counts = {ledger["core"] for ledger in ledgers.values()}
    if len(core_counts) != 1:
        raise ValueError(f"flow core budgets are not matched: {ledgers}")
    return ledgers


def assert_matched_sample_exposure(config: FlowTrainConfig) -> None:
    if config.effective_batch_size != MATCHED_FLOW_EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"all four flows require effective_batch_size={MATCHED_FLOW_EFFECTIVE_BATCH_SIZE}, "
            f"got {config.effective_batch_size}"
        )
    if config.microbatch_size * config.grad_accum_steps != config.effective_batch_size:
        raise ValueError(
            "effective_batch_size must equal microbatch_size*grad_accum_steps; "
            "codec-specific microbatches may not change sample exposure"
        )
