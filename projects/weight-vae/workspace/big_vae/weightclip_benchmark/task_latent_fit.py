from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from big_vae.flow_matching.dataset import GroupedLatentRecord, TileIdentity, canonical_codec_fingerprint
from big_vae.flow_matching.features import FLOW_ARCHITECTURE_FEATURE_SCHEMA
from big_vae.weightclip_benchmark.resnet_functional import ConvSpec, ConvWeightProvider, FunctionalResNet18Slim, FunctionalResNetConfig


class TileDecoder(Protocol):
    """Frozen codec adapter. Implementations retain gradients with respect to codes."""

    def decode_tiles(
        self,
        codes: torch.Tensor,
        *,
        activation_context: torch.Tensor | None,
        architecture_features: torch.Tensor,
        tile_mask: torch.Tensor | None,
        weight_mask: torch.Tensor,
        tile_indices: torch.Tensor,
        layer_key: str,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class LayerTileLayout:
    layer_key: str
    code_indices: tuple[int, ...]
    matrix_shape: tuple[int, int]
    tile_shape: tuple[int, int]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (math.ceil(self.matrix_shape[0] / self.tile_shape[0]), math.ceil(self.matrix_shape[1] / self.tile_shape[1]))

    def validate(self) -> None:
        expected = self.grid_shape[0] * self.grid_shape[1]
        if len(self.code_indices) != expected:
            raise ValueError(f"{self.layer_key}: {len(self.code_indices)} codes for {expected} tiles")
        if len(set(self.code_indices)) != len(self.code_indices):
            raise ValueError(f"{self.layer_key}: duplicate code indices")


class LatentDecodedConvProvider(ConvWeightProvider):
    def __init__(
        self,
        codes: nn.Parameter,
        decoder: TileDecoder,
        layouts: Sequence[LayerTileLayout],
        architecture_features: torch.Tensor,
        tile_mask: torch.Tensor | None,
        *,
        activation_conditioning: bool,
    ) -> None:
        self.codes = codes
        self.decoder = decoder
        self.layouts = {layout.layer_key: layout for layout in layouts}
        self.architecture_features = architecture_features
        self.tile_mask = tile_mask
        self.activation_conditioning = bool(activation_conditioning)
        self.cached_weights: dict[str, torch.Tensor] = {}
        self.cache_active = False
        for layout in layouts:
            layout.validate()

    def get_conv_weight(self, spec: ConvSpec, activation_context: torch.Tensor) -> torch.Tensor:
        if self.cache_active:
            try:
                return self.cached_weights[spec.key]
            except KeyError as error:
                raise RuntimeError(f"static materialization cache is missing {spec.key}") from error
        if spec.key not in self.layouts:
            raise KeyError(f"no code layout for {spec.key}")
        layout = self.layouts[spec.key]
        expected_matrix = (spec.in_channels * spec.kernel_size[0] * spec.kernel_size[1], spec.out_channels)
        if layout.matrix_shape != expected_matrix:
            raise ValueError(f"{spec.key}: layout matrix {layout.matrix_shape}, expected {expected_matrix}")
        index = torch.tensor(layout.code_indices, device=self.codes.device, dtype=torch.long)
        codes = self.codes.index_select(0, index)
        features = self.architecture_features.index_select(0, index)
        mask = None if self.tile_mask is None else self.tile_mask.index_select(0, index)
        rows, cols = layout.grid_shape
        tile_h, tile_w = layout.tile_shape
        contexts: list[torch.Tensor] = []
        weight_masks: list[torch.Tensor] = []
        for tile_index in range(rows * cols):
            row, col = divmod(tile_index, cols)
            valid_h = min(tile_h, layout.matrix_shape[0] - row * tile_h)
            valid_w = min(tile_w, layout.matrix_shape[1] - col * tile_w)
            context = activation_context.new_zeros((activation_context.shape[0], tile_h))
            context[:, :valid_h] = activation_context[:, row * tile_h : row * tile_h + valid_h]
            contexts.append(context)
            weight_mask = torch.zeros((tile_h, tile_w), dtype=torch.bool, device=activation_context.device)
            weight_mask[:valid_h, :valid_w] = True
            weight_masks.append(weight_mask)
        decoded = self.decoder.decode_tiles(
            codes,
            activation_context=torch.stack(contexts) if self.activation_conditioning else None,
            architecture_features=features,
            tile_mask=mask,
            weight_mask=torch.stack(weight_masks),
            tile_indices=index,
            layer_key=spec.key,
        )
        if tuple(decoded.shape) != (rows * cols, tile_h, tile_w):
            raise ValueError(f"decoder returned {tuple(decoded.shape)} for {spec.key}, expected {(rows * cols, tile_h, tile_w)}")
        matrix_rows = [torch.cat([decoded[r * cols + c] for c in range(cols)], dim=1) for r in range(rows)]
        matrix = torch.cat(matrix_rows, dim=0)[: layout.matrix_shape[0], : layout.matrix_shape[1]]
        weight = matrix.transpose(0, 1).reshape(spec.weight_shape).contiguous()
        self.cached_weights[spec.key] = weight
        return weight

    def begin_materialization(self) -> None:
        self.cached_weights.clear()
        self.cache_active = False

    def finish_materialization(self) -> None:
        if set(self.cached_weights) != set(self.layouts):
            missing = sorted(set(self.layouts) - set(self.cached_weights))
            raise RuntimeError(f"materialization did not visit all layers: {missing}")
        self.cache_active = True


@dataclass(slots=True)
class TaskLatentFitConfig:
    steps: int = 500
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    validation_interval: int = 25
    log_interval: int = 10
    device: str = "cuda"
    dtype: str = "float32"
    activation_conditioning: bool = True
    head_policy: str = "original_frozen_source"
    bn_policy: str = "original_frozen_source"
    head_policy_pending_confirmation: bool = False
    seed: int = 0
    task_batch_size: int = 128
    optimization_batches_per_step: int = 1
    output_dir: str = "artifacts/weightclip_benchmark/task_latents"
    model_dropout: float = 0.15


@dataclass(slots=True)
class TaskLatentFitResult:
    initial_validation_loss: float
    best_validation_loss: float
    final_validation_loss: float
    best_step: int
    train_history: list[dict[str, float]]
    output_path: Path | None

    @property
    def improved(self) -> bool:
        return self.best_validation_loss < self.initial_validation_loss


def _resolve_dtype(name: str) -> torch.dtype:
    table = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"unsupported dtype {name!r}")
    return table[name]


_MATCHED_TASK_FIT_FIELDS = (
    "steps",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "validation_interval",
    "log_interval",
    "device",
    "dtype",
    "seed",
    "task_batch_size",
    "optimization_batches_per_step",
)


def task_fit_protocol(config: Any) -> dict[str, Any]:
    return {field: getattr(config, field) for field in _MATCHED_TASK_FIT_FIELDS}


def assert_task_fit_protocol_parity(ours: Any, weightclip: Any) -> None:
    ours_protocol = task_fit_protocol(ours)
    weightclip_protocol = task_fit_protocol(weightclip)
    if ours_protocol != weightclip_protocol:
        raise ValueError(f"ours/WeightCLIP z_task optimizer protocols differ: {ours_protocol} vs {weightclip_protocol}")


def context_tensor_sha256(images: torch.Tensor) -> str:
    value = images.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def write_json_immutable(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable JSON artifact conflict: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return path


def validate_context_provenance(provenance: Mapping[str, Any], images: torch.Tensor | None = None) -> None:
    if provenance.get("source_split") != "train" or not bool(provenance.get("manifest_recorded", False)):
        raise ValueError("decoder context must come from a manifest-recorded train-only image pool")
    missing = [key for key in ("pool_id", "manifest_sha256", "image_indices") if not provenance.get(key)]
    if missing:
        raise ValueError(f"decoder context provenance is missing manifest fields: {missing}")
    if images is not None and provenance.get("images_sha256") != context_tensor_sha256(images):
        raise ValueError("decoder context image tensor does not match its manifest-recorded SHA-256")


def _batch_xy(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        x = batch.get("images", batch.get("x"))
        y = batch.get("labels", batch.get("y"))
    else:
        x, y = batch[:2]
    if x is None or y is None:
        raise ValueError("task batch must contain images and labels")
    return x.to(device=device, non_blocking=True), y.to(device=device, dtype=torch.long, non_blocking=True)


@torch.no_grad()
def evaluate_task_loss(model: nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        images, labels = _batch_xy(batch, device)
        logits = model(images)
        total_loss += float(F.cross_entropy(logits.float(), labels, reduction="sum").item())
        total_examples += int(labels.numel())
    if total_examples == 0:
        raise ValueError("validation loader is empty")
    return total_loss / total_examples


def fit_task_latents(
    *,
    initial_codes: torch.Tensor,
    decoder: TileDecoder,
    layouts: Sequence[LayerTileLayout],
    architecture_features: torch.Tensor,
    tile_mask: torch.Tensor | None,
    source_state: Mapping[str, torch.Tensor],
    model_config: FunctionalResNetConfig,
    train_loader: Any,
    validation_loader: Any,
    config: TaskLatentFitConfig,
    identity: Mapping[str, str],
    dataset_embedding: torch.Tensor,
    dataset_embedding_bank: torch.Tensor,
    dataset_embedding_bank_provenance: Mapping[str, Any],
    tile_identities: Sequence[TileIdentity],
    context_images: torch.Tensor,
    context_provenance: Mapping[str, Any],
    fit_protocol_provenance: Mapping[str, Any] | None = None,
) -> tuple[GroupedLatentRecord, TaskLatentFitResult]:
    """Jointly optimize every body code of one checkpoint on native task CE."""

    if config.head_policy_pending_confirmation:
        raise ValueError("z_task critic head policy is resolved; pending_confirmation must be false")
    if config.head_policy != "original_frozen_source" or config.bn_policy != "original_frozen_source":
        raise ValueError("z_task critic must use the frozen original source head and BN")
    if config.optimization_batches_per_step != 1:
        raise ValueError("matched z_task budget requires exactly one task minibatch per optimizer step")
    if config.model_dropout != 0.15 or model_config.dropout != config.model_dropout:
        raise ValueError("z_task critic dropout must exactly match the zoo/evaluation contract 0.15")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dtype = _resolve_dtype(config.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if len(tile_identities) != initial_codes.shape[0]:
        raise ValueError("tile identities do not align with initial codes")
    validate_context_provenance(context_provenance, context_images)
    used_indices = sorted(index for layout in layouts for index in layout.code_indices)
    if used_indices != list(range(initial_codes.shape[0])):
        raise ValueError("every latent must appear exactly once in the functional graph")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "task_latent_fit_start",
                "config": asdict(config),
                "identity": dict(identity),
                "device": str(device),
                "dtype": str(dtype),
                "output_dir": str(output_dir.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    decoder_module = decoder if isinstance(decoder, nn.Module) else None
    if decoder_module is not None:
        decoder_module.to(device=device)
        decoder_module.eval()
        decoder_module.requires_grad_(False)
    codes = nn.Parameter(initial_codes.detach().to(device=device, dtype=dtype).clone())
    features = architecture_features.to(device=device, dtype=dtype)
    mask = None if tile_mask is None else tile_mask.to(device=device, dtype=torch.bool)
    provider = LatentDecodedConvProvider(
        codes,
        decoder,
        layouts,
        features,
        mask,
        activation_conditioning=config.activation_conditioning,
    )
    resolved_model_config = copy.copy(model_config)
    resolved_model_config.head_policy = config.head_policy
    resolved_model_config.bn_policy = config.bn_policy
    model = FunctionalResNet18Slim(
        resolved_model_config,
        provider,
        source_state={key: value.to(device=device) for key, value in source_state.items()},
        random_head_seed=config.seed,
    ).to(device=device)
    optimizer = torch.optim.AdamW([codes], lr=config.learning_rate, weight_decay=config.weight_decay)
    fixed_context_images = context_images.to(device=device, non_blocking=True)

    decoder_calls = 0

    def materialize() -> None:
        nonlocal decoder_calls
        provider.begin_materialization()
        model(fixed_context_images)
        provider.finish_materialization()
        decoder_calls += len(layouts)

    with torch.no_grad():
        materialize()
    initial_loss = evaluate_task_loss(model, validation_loader, device)
    best_loss = initial_loss
    best_codes = codes.detach().cpu().clone()
    best_step = 0
    history: list[dict[str, float]] = []
    iterator = iter(train_loader)
    started = time.monotonic()
    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        images, labels = _batch_xy(batch, device)
        optimizer.zero_grad(set_to_none=True)
        materialize()
        logits = model(images)
        loss = F.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite task loss at step {step}")
        loss.backward()
        if codes.grad is None or not torch.isfinite(codes.grad).all():
            raise FloatingPointError(f"missing/non-finite latent gradients at step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([codes], config.grad_clip).item())
        optimizer.step()
        row = {"step": float(step), "train_loss": float(loss.item()), "grad_norm": grad_norm, "elapsed_sec": time.monotonic() - started}
        if step % config.validation_interval == 0 or step == config.steps:
            with torch.no_grad():
                materialize()
            val_loss = evaluate_task_loss(model, validation_loader, device)
            row["validation_loss"] = val_loss
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                best_codes = codes.detach().cpu().clone()
        history.append(row)
        if step % config.log_interval == 0 or step == 1 or step == config.steps:
            print(json.dumps({"stage": "task_latent_fit", **row, "best_validation_loss": best_loss}), flush=True)
    with torch.no_grad():
        codes.copy_(best_codes.to(device=device, dtype=dtype))
        materialize()
    final_loss = evaluate_task_loss(model, validation_loader, device)
    improved = bool(
        math.isfinite(initial_loss)
        and math.isfinite(best_loss)
        and math.isfinite(final_loss)
        and best_loss < initial_loss
    )
    if not improved:
        failure_dir = output_dir / "failed"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"{identity['group_id']}.metrics.json"
        write_json_immutable(
            failure_path,
            {
                    "fit_status": "failed",
                    "initial_validation_loss": initial_loss,
                    "best_validation_loss": best_loss,
                    "final_validation_loss": final_loss,
                    "best_step": best_step,
                    "history": history,
                    "reason": "best_validation_loss_did_not_strictly_improve",
                "decoder_calls": decoder_calls,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            },
        )
        raise RuntimeError(
            f"task-fit gate failed: best validation loss {best_loss:.6g} "
            f"did not beat z_enc {initial_loss:.6g}; failure={failure_path.resolve()}"
        )
    record = GroupedLatentRecord(
        group_id=str(identity["group_id"]),
        dataset_id=str(identity["dataset_id"]),
        lineage_id=str(identity["lineage_id"]),
        checkpoint_id=str(identity["checkpoint_id"]),
        split=str(identity["split"]),
        codec=str(identity["codec"]),
        codec_fingerprint=canonical_codec_fingerprint(str(identity["codec"]), decoder.provenance),
        identities=tuple(tile_identities),
        anchor_identities=tuple(tile_identities),
        z_enc=initial_codes.detach().cpu(),
        z_task=best_codes,
        dataset_embedding=dataset_embedding.detach().cpu(),
        dataset_embedding_bank=dataset_embedding_bank.detach().cpu(),
        architecture_features=architecture_features.detach().cpu(),
        tile_mask=None if tile_mask is None else tile_mask.detach().cpu(),
        provenance={
            "fit_status": "success",
            "task_latent_fit_config": asdict(config),
            "head_policy": config.head_policy,
            "head_policy_pending_confirmation": config.head_policy_pending_confirmation,
            "bn_policy": config.bn_policy,
            "initial_validation_loss": initial_loss,
            "best_validation_loss": best_loss,
            "best_step": best_step,
            "context_provenance": dict(context_provenance),
            "dataset_embedding_bank_provenance": dict(dataset_embedding_bank_provenance),
            "fit_protocol_provenance": dict(fit_protocol_provenance or {}),
            "architecture_conditioning": {
                "schema": list(FLOW_ARCHITECTURE_FEATURE_SCHEMA),
                "granularity": "128x128_operator_tile_repeated_over_ae_slots",
                "operator_matrix_orientation": "[in_times_kernel,out]",
            },
        },
    )
    output_path = record.save(output_dir / f"{record.group_id}.pt")
    metrics_path = output_dir / f"{record.group_id}.metrics.json"
    write_json_immutable(
        metrics_path,
        {
                "initial_validation_loss": initial_loss,
                "best_validation_loss": best_loss,
                "final_validation_loss": final_loss,
                "best_step": best_step,
                "fit_status": "success",
                "improved": True,
                "history": history,
                "record_path": str(output_path.resolve()),
                "decoder_calls": decoder_calls,
                "elapsed_sec": time.monotonic() - started,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "optimizer": "AdamW",
            },
    )
    result = TaskLatentFitResult(initial_loss, best_loss, final_loss, best_step, history, output_path)
    print(
        json.dumps(
            {
                "stage": "task_latent_fit_complete",
                "record_path": str(output_path.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "initial_validation_loss": initial_loss,
                "best_validation_loss": best_loss,
                "improved": result.improved,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record, result
