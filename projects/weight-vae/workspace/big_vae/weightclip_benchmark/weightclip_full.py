from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.func import functional_call

from big_vae.flow_matching.dataset import GroupedLatentRecord, TileIdentity, canonical_codec_fingerprint
from big_vae.flow_matching.features import FLOW_ARCHITECTURE_FEATURE_SCHEMA

from .decoder_adapters import OfficialWeightCLIPMultiWindowDecoderAdapter
from .resnet18slim import ResNet18Slim
from .task_latent_fit import _resolve_dtype, validate_context_provenance, write_json_immutable


@dataclass(slots=True)
class WeightCLIPTaskFitConfig:
    steps: int = 500
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    validation_interval: int = 25
    log_interval: int = 10
    grad_clip: float = 5.0
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0
    output_dir: str = "artifacts/weightclip_benchmark/task_latents/weightclip"
    head_policy: str = "original_frozen_source"
    bn_policy: str = "original_frozen_source"
    task_batch_size: int = 128
    optimization_batches_per_step: int = 1
    activation_conditioning: bool = False
    head_policy_pending_confirmation: bool = False
    model_dropout: float = 0.15


def compose_controlled_code(z_enc: torch.Tensor, delta: torch.Tensor, body_token_mask: torch.Tensor) -> torch.Tensor:
    """Expose full official windows while giving optimization DOF only to body rows."""

    if z_enc.shape != delta.shape or tuple(body_token_mask.shape) != tuple(z_enc.shape[:2]):
        raise ValueError("controlled code tensors/mask do not align")
    return z_enc + delta * body_token_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)


def _fixed_policy_state(
    decoded: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    config: WeightCLIPTaskFitConfig,
    fit_protocol_provenance: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    state = dict(decoded)
    if config.head_policy != "original_frozen_source" or config.bn_policy != "original_frozen_source":
        raise ValueError("first WeightCLIP task-fit path supports only frozen original critic head/BN")
    for key, value in source.items():
        is_head = key.startswith("fc.")
        is_bn_buffer = (key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked")) and (
            key.startswith("bn1.") or ".bn" in key or ".shortcut.1." in key
        )
        if is_head or is_bn_buffer:
            state[key] = value.detach().to(next(iter(decoded.values())).device)
    return state


def _loss_for_batches(
    template: ResNet18Slim,
    state: Mapping[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros((), device=device)
    count = 0
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        logits = functional_call(template, state, (images,))
        total = total + F.cross_entropy(logits, labels, reduction="sum")
        count += int(labels.numel())
    return total / max(count, 1)


def _cycle_train_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Repeat a re-iterable minibatch source, failing on an empty epoch."""

    while True:
        seen = False
        for batch in batches:
            seen = True
            yield batch
        if not seen:
            raise ValueError("WeightCLIP task-fit train loader is empty")


def fit_weightclip_multiwindow_task_latent(
    *,
    initial_code: torch.Tensor,
    decoder: OfficialWeightCLIPMultiWindowDecoderAdapter,
    source_state: Mapping[str, torch.Tensor],
    train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    context_provenance: Mapping[str, Any],
    identity: Mapping[str, str],
    dataset_embedding: torch.Tensor,
    dataset_embedding_bank: torch.Tensor,
    dataset_embedding_bank_provenance: Mapping[str, Any],
    architecture_features: torch.Tensor,
    config: WeightCLIPTaskFitConfig,
    fit_protocol_provenance: Mapping[str, Any] | None = None,
) -> GroupedLatentRecord:
    validate_context_provenance(context_provenance)
    if config.head_policy_pending_confirmation:
        raise ValueError("z_task critic head policy is resolved; pending_confirmation must be false")
    if config.head_policy != "original_frozen_source" or config.bn_policy != "original_frozen_source":
        raise ValueError("z_task critic must use the frozen original source head and BN")
    if config.optimization_batches_per_step != 1:
        raise ValueError("matched z_task budget requires exactly one task minibatch per optimizer step")
    if config.model_dropout != 0.15:
        raise ValueError("z_task critic dropout must exactly match the zoo/evaluation contract 0.15")
    expected = (decoder.window_count, decoder.window_size)
    if initial_code.ndim != 3 or tuple(initial_code.shape[:2]) != expected:
        raise ValueError(f"WeightCLIP task fitting requires grouped official windows {expected}, got {tuple(initial_code.shape)}")
    device = torch.device(config.device)
    dtype = _resolve_dtype(config.dtype)
    torch.manual_seed(config.seed)
    decoder.to(device).eval().requires_grad_(False)
    z_enc = initial_code.detach().to(device=device, dtype=dtype).clone()
    delta = torch.nn.Parameter(torch.zeros_like(z_enc))
    num_classes = int(source_state["fc.weight"].shape[0])
    width = float(source_state["conv1.weight"].shape[0]) / 64.0
    template = ResNet18Slim(o_dim=num_classes, width_mult=width, dropout=config.model_dropout, init_type=None).to(device).eval()
    optimizer = torch.optim.AdamW([delta], lr=config.learning_rate, weight_decay=config.weight_decay)

    decoder_calls = 0

    def decode() -> dict[str, torch.Tensor]:
        nonlocal decoder_calls
        decoder_calls += 1
        code = compose_controlled_code(z_enc, delta, decoder.body_token_mask)
        return _fixed_policy_state(decoder.decode_state(code), source_state, config)

    with torch.no_grad():
        initial_validation = float(_loss_for_batches(template, decode(), validation_batches, device).item())
    best_loss, best_step = initial_validation, 0
    best_delta = delta.detach().cpu().clone()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "weightclip_multiwindow_task_fit_start", "config": asdict(config), "initial_validation_loss": initial_validation}), flush=True)
    train_iterator = iter(_cycle_train_batches(train_batches))
    history: list[dict[str, float]] = []
    started = time.monotonic()
    for step in range(1, config.steps + 1):
        images, labels = next(train_iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_for_batches(template, decode(), [(images, labels)], device)
        loss.backward()
        if delta.grad is None:
            raise RuntimeError("controlled WeightCLIP task fit produced no latent gradient")
        if torch.count_nonzero(delta.grad[~decoder.body_token_mask]).item() != 0:
            raise RuntimeError("nonbody WeightCLIP latent rows received a gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([delta], config.grad_clip).item())
        optimizer.step()
        row = {
            "step": float(step),
            "train_loss": float(loss.item()),
            "grad_norm": grad_norm,
            "elapsed_sec": time.monotonic() - started,
        }
        if step % config.validation_interval == 0 or step == config.steps:
            with torch.no_grad():
                validation = float(_loss_for_batches(template, decode(), validation_batches, device).item())
            row["validation_loss"] = validation
            if validation < best_loss:
                best_loss, best_step, best_delta = validation, step, delta.detach().cpu().clone()
        history.append(row)
        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(json.dumps({"stage": "weightclip_multiwindow_task_fit", **row, "best_validation_loss": best_loss}), flush=True)
    with torch.no_grad():
        delta.copy_(best_delta.to(device=device, dtype=dtype))
        final_loss = float(_loss_for_batches(template, decode(), validation_batches, device).item())
    identities = tuple(
        TileIdentity(
            str(identity["dataset_id"]),
            str(identity["lineage_id"]),
            str(identity["checkpoint_id"]),
            "__weightclip_official_window__",
            window_index,
            0,
        )
        for window_index in range(decoder.window_count)
    )
    if tuple(architecture_features.shape[:2]) != (decoder.window_count, decoder.window_size):
        raise ValueError("architecture_features must contain shared token features for every official window token")
    if not (
        math.isfinite(initial_validation)
        and math.isfinite(best_loss)
        and math.isfinite(final_loss)
        and best_loss < initial_validation
    ):
        failure_dir = output_dir / "failed"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"{identity['group_id']}.metrics.json"
        write_json_immutable(
            failure_path,
            {
                    "fit_status": "failed",
                    "initial_validation_loss": initial_validation,
                    "best_validation_loss": best_loss,
                    "final_validation_loss": final_loss,
                    "best_step": best_step,
                    "history": history,
                    "reason": "best_validation_loss_did_not_strictly_improve",
                    "decoder_calls": decoder_calls,
                    "elapsed_sec": time.monotonic() - started,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                },
        )
        raise RuntimeError(
            f"WeightCLIP multi-window z_task failed to improve source validation loss; "
            f"failure={failure_path.resolve()}"
        )
    record = GroupedLatentRecord(
        group_id=str(identity["group_id"]),
        dataset_id=str(identity["dataset_id"]),
        lineage_id=str(identity["lineage_id"]),
        checkpoint_id=str(identity["checkpoint_id"]),
        split=str(identity["split"]),
        codec="weightclip",
        codec_fingerprint=canonical_codec_fingerprint("weightclip", decoder.provenance),
        identities=identities,
        anchor_identities=identities,
        z_enc=initial_code.detach().cpu(),
        z_task=compose_controlled_code(initial_code.detach().cpu(), best_delta, decoder.body_token_mask.detach().cpu()),
        dataset_embedding=dataset_embedding.detach().cpu(),
        dataset_embedding_bank=dataset_embedding_bank.detach().cpu(),
        architecture_features=architecture_features.detach().cpu(),
        tile_mask=decoder.window_token_mask.detach().cpu(),
        provenance={
            "fit_status": "success",
            "task_fit_config": asdict(config),
            "initial_validation_loss": initial_validation,
            "best_validation_loss": best_loss,
            "final_validation_loss": final_loss,
            "best_step": best_step,
            "context_provenance": dict(context_provenance),
            "decoder_provenance": decoder.provenance,
            "dataset_embedding_bank_provenance": dict(dataset_embedding_bank_provenance),
            "fit_protocol_provenance": dict(fit_protocol_provenance or {}),
            "architecture_conditioning": {
                "schema": list(FLOW_ARCHITECTURE_FEATURE_SCHEMA),
                "granularity": "official_sparse_token_first_scalar_conceptual_tile",
                "known_deviation": "one_sparse_token_may_span_multiple_conceptual_input_tiles",
                "operator_matrix_orientation": "[in_times_kernel,out]",
            },
            "rate_ledger": decoder.rate_ledger(initial_code.shape[-1]),
            "controlled_body_policy": "conv_and_bn_affine; source_head_and_bn_running_buffers; full_windows_decoder_context",
        },
    )
    output_path = record.save(output_dir / f"{record.group_id}.pt")
    write_json_immutable(
        output_dir / f"{record.group_id}.metrics.json",
        {
            "fit_status": "success",
            "initial_validation_loss": initial_validation,
            "best_validation_loss": best_loss,
            "final_validation_loss": final_loss,
            "best_step": best_step,
            "improved": True,
            "history": history,
            "record_path": str(output_path.resolve()),
            "decoder_calls": decoder_calls,
            "elapsed_sec": time.monotonic() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "optimizer": "AdamW",
        },
    )
    return record


def _reset_head(model: ResNet18Slim, seed: int) -> None:
    devices = [model.fc.weight.device] if model.fc.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        model.fc.reset_parameters()


def _calibrate_bn(model: nn.Module, batches: list[torch.Tensor] | None) -> None:
    if not batches:
        return
    model.train()
    with torch.no_grad():
        for images in batches:
            model(images.to(next(model.parameters()).device))
    model.eval()


def materialize_weightclip_controlled_model(
    decoder: OfficialWeightCLIPMultiWindowDecoderAdapter,
    code: torch.Tensor,
    *,
    num_classes: int,
    width_mult: float,
    dropout: float = 0.15,
    paired_head_seed: int = 0,
    bn_calibration_batches: list[torch.Tensor] | None = None,
) -> ResNet18Slim:
    with torch.no_grad():
        state = decoder.decode_state(code)
    model = ResNet18Slim(o_dim=num_classes, width_mult=width_mult, dropout=dropout, init_type=None).to(code.device)
    target = model.state_dict()
    # Controlled policy: decoded conv + BN affine, fresh paired head, reset BN running state.
    for key, value in state.items():
        is_conv = key == "conv1.weight" or ".conv" in key or ".shortcut.0.weight" in key
        is_bn_affine = (key.endswith(".weight") or key.endswith(".bias")) and (
            key.startswith("bn1.") or ".bn" in key or ".shortcut.1." in key
        )
        if key in target and (is_conv or is_bn_affine):
            target[key] = value.to(target[key])
    model.load_state_dict(target, strict=True)
    _reset_head(model, paired_head_seed)
    _calibrate_bn(model, bn_calibration_batches)
    return model


def materialize_weightclip_native_model(
    decoder: OfficialWeightCLIPMultiWindowDecoderAdapter,
    code: torch.Tensor,
    *,
    num_classes: int,
    width_mult: float,
    dropout: float = 0.15,
) -> ResNet18Slim:
    """Native audit path: retain the released decoded head and complete BN state."""

    with torch.no_grad():
        state = decoder.decode_state(code)
    model = ResNet18Slim(o_dim=num_classes, width_mult=width_mult, dropout=dropout, init_type=None).to(code.device)
    model.load_state_dict({key: value.to(model.state_dict()[key]) for key, value in state.items()}, strict=True)
    return model
