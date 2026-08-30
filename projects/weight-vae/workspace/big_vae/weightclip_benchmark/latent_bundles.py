"""Manifest-driven Stage-D bundles for the two frozen codec substrates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from big_vae.datasets.operator_bank import OperatorBankTrainingDataset, OperatorBankSample
from big_vae.weightclip_benchmark.activation_capture import (
    capture_native_train_activations,
    context_tiles,
    stable_seed,
)
from big_vae.flow_matching.dataset import TileIdentity, canonical_codec_fingerprint
from big_vae.flow_matching.features import FLOW_ARCHITECTURE_FEATURE_SCHEMA, semantic_token_features
from big_vae.weightclip_benchmark.dataset_payload import load_cached_dataset_payload
from big_vae.weightclip_benchmark.decoder_adapters import (
    BigVAETileDecoderAdapter,
    OfficialWeightCLIPMultiWindowDecoderAdapter,
    build_bigvae_tile_decoder_adapter,
)
from big_vae.weightclip_benchmark.manifests import sha256_file, write_json_immutable
from big_vae.weightclip_benchmark.parameter_adapters import (
    parameter_to_matrix,
    supported_operator_specs,
    tile_matrix,
)
from big_vae.weightclip_benchmark.resnet18slim import ResNet18Slim
from big_vae.weightclip_benchmark.task_latent_fit import LayerTileLayout, context_tensor_sha256


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def create(cls, path: str | Path) -> "ArtifactRef":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return cls(str(resolved), sha256_file(resolved), resolved.stat().st_size)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(str(payload["path"]), str(payload["sha256"]), int(payload["bytes"]))

    def verify(self, label: str) -> Path:
        path = Path(self.path).resolve()
        if not path.is_file() or path.stat().st_size != self.bytes or sha256_file(path) != self.sha256:
            raise ValueError(f"{label} provenance mismatch: {path}")
        return path


def provenance_sha256(payload: Mapping[str, Any]) -> str:
    """Stable hash for JSON-compatible resolved codec/config provenance."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_decoder_provenance_matches_bundle(
    loaded: Mapping[str, Any],
    bundled: Mapping[str, Any],
    *,
    codec: str,
) -> None:
    """Reject optimization under any codec checkpoint/config other than bundle build."""

    if loaded.get("codec") != codec or bundled.get("codec") != codec:
        raise ValueError(f"{codec} decoder codec provenance mismatch")
    if loaded.get("checkpoint_sha256") != bundled.get("checkpoint_sha256"):
        raise ValueError(f"{codec} decoder checkpoint hash disagrees with task bundle")
    if codec == "ours":
        if loaded.get("model_config_sha256") != bundled.get("model_config_sha256"):
            raise ValueError("ours decoder model config hash disagrees with task bundle")
        return
    if codec != "weightclip":
        raise ValueError(f"unsupported codec provenance {codec!r}")
    loaded_official = loaded.get("official", {})
    bundled_official = bundled.get("official", {})
    for label in ("checkpoint", "dataset_encoder"):
        if loaded_official.get(label, {}).get("sha256") != bundled_official.get(label, {}).get("sha256"):
            raise ValueError(f"WeightCLIP official {label} hash disagrees with task bundle")
    if loaded_official.get("contract_fingerprint") != bundled_official.get("contract_fingerprint"):
        raise ValueError("WeightCLIP official contract fingerprint disagrees with task bundle")
    if provenance_sha256(loaded_official.get("config", {})) != provenance_sha256(
        bundled_official.get("config", {})
    ):
        raise ValueError("WeightCLIP resolved codec config hash disagrees with task bundle")


def load_sealed_ours_decoder(seal_ref: ArtifactRef, *, device: str) -> BigVAETileDecoderAdapter:
    """Resolve the immutable deterministic-AE seal without guessing paths."""

    seal_path = seal_ref.verify("ours codec seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    required = {"schema_version", "codec", "deterministic_ae", "tile_shape", "checkpoint", "model_config"}
    if required - set(seal):
        raise ValueError(f"ours codec seal is missing {sorted(required - set(seal))}")
    if int(seal["schema_version"]) != 1 or seal["codec"] != "ours" or seal["deterministic_ae"] is not True:
        raise ValueError("ours codec seal is not a deterministic AE v1 seal")
    if tuple(seal["tile_shape"]) != (128, 128):
        raise ValueError("ours codec seal must bind exact 128x128 tiles")
    checkpoint = ArtifactRef.from_mapping(seal["checkpoint"])
    model_config = ArtifactRef.from_mapping(seal["model_config"])
    checkpoint.verify("ours AE checkpoint")
    model_config.verify("ours AE model config")
    decoder = build_bigvae_tile_decoder_adapter(
        checkpoint_path=checkpoint.path,
        model_config_path=model_config.path,
        device=device,
    )
    if decoder.provenance["checkpoint_sha256"] != checkpoint.sha256:
        raise ValueError("ours decoder checkpoint disagrees with codec seal")
    if decoder.provenance["model_config_sha256"] != model_config.sha256:
        raise ValueError("ours decoder model config disagrees with codec seal")
    return decoder


def load_dataset_pt(reference: ArtifactRef) -> dict[str, Any]:
    path = reference.verify("dataset.pt")
    payload = load_cached_dataset_payload(str(path))
    required = {"trainset", "valset", "testset"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"dataset.pt must contain exactly {sorted(required)}")
    for split, dataset in payload.items():
        data, targets = getattr(dataset, "data", None), getattr(dataset, "targets", None)
        if not torch.is_tensor(data) or not torch.is_tensor(targets):
            raise TypeError(f"dataset.pt {split} is missing tensor data/targets")
        if data.ndim != 4 or tuple(data.shape[1:]) != (3, 32, 32) or targets.ndim != 1:
            raise ValueError(f"dataset.pt {split} has invalid shapes {tuple(data.shape)}/{tuple(targets.shape)}")
        if len(data) != len(targets) or not torch.isfinite(data).all():
            raise ValueError(f"dataset.pt {split} is invalid")
        if float(data.min()) < -1.0001 or float(data.max()) > 1.0001:
            raise ValueError(f"dataset.pt {split} is not official [-1,1] normalized")
    return payload


def load_train_dataset_pt(reference: ArtifactRef) -> dict[str, Any]:
    """Load and validate only the target train split before the OOD unseal."""

    path = reference.verify("dataset.pt")
    payload = load_cached_dataset_payload(str(path), splits=("trainset",))
    if not isinstance(payload, dict) or "trainset" not in payload:
        raise ValueError("dataset.pt is missing trainset")
    dataset = payload["trainset"]
    data, targets = getattr(dataset, "data", None), getattr(dataset, "targets", None)
    if not torch.is_tensor(data) or not torch.is_tensor(targets):
        raise TypeError("dataset.pt trainset is missing tensor data/targets")
    if data.ndim != 4 or tuple(data.shape[1:]) != (3, 32, 32) or targets.ndim != 1:
        raise ValueError(f"dataset.pt trainset has invalid shapes {tuple(data.shape)}/{tuple(targets.shape)}")
    if len(data) != len(targets) or not torch.isfinite(data).all():
        raise ValueError("dataset.pt trainset is invalid")
    if float(data.min()) < -1.0001 or float(data.max()) > 1.0001:
        raise ValueError("dataset.pt trainset is not official [-1,1] normalized")
    return {"trainset": dataset}


def load_checkpoint_state(reference: ArtifactRef) -> dict[str, torch.Tensor]:
    payload = torch.load(reference.verify("source checkpoint"), map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model", payload)) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError("source checkpoint has no state mapping")
    result = {str(key).removeprefix("module."): value.detach().cpu().float().contiguous() for key, value in state.items()}
    required = {"conv1.weight", "fc.weight", "fc.bias", "bn1.weight", "bn1.running_mean"}
    if not required <= set(result):
        raise ValueError(f"source checkpoint is not ResNet18Slim; missing {sorted(required - set(result))}")
    return result


def encode_dataset_embedding(encoder: Any, train_images: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if len(indices) != 10 or len(set(int(index) for index in indices)) != 10:
        raise ValueError("WeightCLIP dataset embedding requires ten distinct train indices")
    if min(indices) < 0 or max(indices) >= len(train_images):
        raise IndexError("dataset prompt index is outside the train split")
    prompt = train_images[list(indices)].unsqueeze(0)
    with torch.inference_mode():
        if hasattr(encoder, "encode"):
            output = encoder.encode(prompt)
        elif hasattr(encoder, "get_embeddings"):
            output = encoder.get_embeddings(prompt)
        else:
            output = encoder(prompt)
    if not torch.is_tensor(output) or output.shape[0] != 1:
        raise ValueError("dataset encoder did not return one embedding")
    embedding = output.reshape(1, -1)[0].detach().cpu().float()
    if embedding.ndim != 1 or not torch.isfinite(embedding).all():
        raise ValueError("dataset embedding is invalid")
    return embedding


def encode_dataset_embedding_bank(
    encoder: Any,
    train_images: torch.Tensor,
    canonical_indices: Sequence[int],
    *,
    candidate_count: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a sealed deterministic bank of distinct ten-image train prompts."""

    if candidate_count <= 0 or len(train_images) < 10:
        raise ValueError("prompt candidate bank requires positive count and at least ten train images")
    candidates = [tuple(int(index) for index in canonical_indices)]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    seen = {candidates[0]}
    while len(candidates) < candidate_count:
        indices = tuple(sorted(torch.randperm(len(train_images), generator=generator)[:10].tolist()))
        if indices not in seen:
            seen.add(indices)
            candidates.append(indices)
    embeddings = torch.stack([encode_dataset_embedding(encoder, train_images, indices) for indices in candidates])
    index_payload = json.dumps(candidates, separators=(",", ":")).encode()
    tensor_payload = embeddings.detach().cpu().contiguous().numpy().tobytes()
    return embeddings, {
        "schema_version": 1,
        "source_split": "train",
        "candidate_count": len(candidates),
        "images_per_candidate": 10,
        "seed": int(seed),
        "candidate_indices": [list(indices) for indices in candidates],
        "indices_sha256": hashlib.sha256(index_payload).hexdigest(),
        "embedding_tensor_sha256": hashlib.sha256(tensor_payload).hexdigest(),
        "selection": "deterministic_step_microstep_group_sampler_without_statistical_unit_duplication",
    }


def _ours_architecture_features(
    layouts: Sequence[LayerTileLayout],
    state: Mapping[str, torch.Tensor],
    *,
    token_count: int,
) -> torch.Tensor:
    group_count = sum(len(layout.code_indices) for layout in layouts)
    layer_indices = torch.zeros((group_count, token_count), dtype=torch.long)
    tile_rows = torch.zeros_like(layer_indices)
    tile_cols = torch.zeros_like(layer_indices)
    layer_keys: list[str] = []
    matrix_shapes: list[tuple[int, int]] = []
    kernels: list[tuple[int, int]] = []
    strides: list[tuple[int, int]] = []
    for layer_index, layout in enumerate(layouts):
        key = f"{layout.layer_key}.weight"
        if key not in state:
            raise ValueError(f"architecture feature ledger lacks {key}")
        layer_keys.append(key)
        matrix_shapes.append(tuple(int(value) for value in layout.matrix_shape))
        kernel, stride = _kernel_stride(key, state[key])
        kernels.append(kernel)
        strides.append(stride)
        layer_indices[list(layout.code_indices)] = layer_index
        for local_index, code_index in enumerate(layout.code_indices):
            row, col = divmod(local_index, int(layout.grid_shape[1]))
            tile_rows[code_index] = row
            tile_cols[code_index] = col
    valid = torch.ones_like(layer_indices, dtype=torch.bool)
    depths = _canonical_architecture_depths(tuple(layer_keys), state)
    return semantic_token_features(
        valid=valid,
        body=valid,
        layer_indices=layer_indices,
        layer_keys=tuple(layer_keys),
        operator_matrix_shapes=tuple(matrix_shapes),
        kernels=tuple(kernels),
        strides=tuple(strides),
        normalized_depths=depths,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


def _kernel_stride(key: str, value: torch.Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
    if value.ndim != 4:
        return (1, 1), (1, 1)
    kernel = (int(value.shape[2]), int(value.shape[3]))
    normalized = key.removesuffix(".weight")
    downsample = (
        any(f"layer{stage}.0" in normalized for stage in (2, 3, 4))
        and (normalized.endswith(".conv1") or normalized.endswith(".shortcut.0"))
    )
    return kernel, ((2, 2) if downsample else (1, 1))


def _operator_matrix_shape(value: torch.Tensor) -> tuple[int, int]:
    if value.ndim == 4:
        return int(value.shape[1] * value.shape[2] * value.shape[3]), int(value.shape[0])
    if value.ndim == 2:
        return int(value.shape[1]), int(value.shape[0])
    if value.ndim == 1:
        return 1, int(value.shape[0])
    raise ValueError(f"unsupported E1 operator tensor shape {tuple(value.shape)}")


def _canonical_architecture_depths(
    layer_keys: tuple[str, ...],
    state: Mapping[str, torch.Tensor],
) -> tuple[float, ...]:
    conv_keys = tuple(
        key
        for key, value in state.items()
        if key.endswith(".weight")
        and value.ndim == 4
        and (key == "conv1.weight" or ".conv" in key or ".shortcut.0.weight" in key)
    )
    if not conv_keys:
        raise ValueError("canonical architecture depth ledger found no convolution operators")
    conv_depth = {key: index / max(len(conv_keys) - 1, 1) for index, key in enumerate(conv_keys)}

    def associated_conv(key: str) -> str | None:
        if key.startswith("bn1."):
            return "conv1.weight"
        if ".shortcut.1." in key:
            return key.replace(".shortcut.1.", ".shortcut.0.").replace("bias", "weight")
        if ".bn1." in key:
            return key.replace(".bn1.", ".conv1.").replace("bias", "weight")
        if ".bn2." in key:
            return key.replace(".bn2.", ".conv2.").replace("bias", "weight")
        return None

    depths: list[float] = []
    for key in layer_keys:
        if key in conv_depth:
            depths.append(conv_depth[key])
        elif key.startswith("fc."):
            depths.append(1.0)
        else:
            linked = associated_conv(key)
            depths.append(conv_depth.get(linked, 1.0))
    return tuple(depths)


def weightclip_window_architecture_features(decoder: OfficialWeightCLIPMultiWindowDecoderAdapter) -> torch.Tensor:
    keys = tuple(decoder.ordered_weight_keys)
    matrix_shapes = tuple(_operator_matrix_shape(decoder.anchor_state[key]) for key in keys)
    kernel_stride = tuple(_kernel_stride(key, decoder.anchor_state[key]) for key in keys)
    depths = _canonical_architecture_depths(keys, decoder.anchor_state)
    valid = decoder.window_token_mask.detach().cpu()
    positions = decoder.anchor_pos.detach().cpu().long()
    layer_indices = positions[..., 1]
    tile_rows = torch.zeros_like(layer_indices)
    tile_cols = torch.zeros_like(layer_indices)
    token_scalar_width = int(decoder.anchor_tokens.shape[-1])
    for layer_index in range(len(keys)):
        for channel in torch.unique(positions[..., 2][valid & (layer_indices == layer_index)]).tolist():
            selected = valid & (layer_indices == layer_index) & (positions[..., 2] == int(channel))
            global_ids = positions[..., 0][selected]
            if global_ids.numel() == 0:
                continue
            chunk = global_ids - global_ids.min()
            tile_rows[selected] = (chunk * token_scalar_width) // 128
            tile_cols[selected] = int(channel) // 128
    return semantic_token_features(
        valid=valid,
        body=decoder.body_token_mask.detach().cpu(),
        layer_indices=layer_indices,
        layer_keys=keys,
        operator_matrix_shapes=matrix_shapes,
        kernels=tuple(item[0] for item in kernel_stride),
        strides=tuple(item[1] for item in kernel_stride),
        normalized_depths=depths,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


def create_ood_template_checkpoint(
    output_path: str | Path,
    *,
    num_classes: int,
    width_mult: float = 0.5,
    seed: int = 0,
) -> ArtifactRef:
    """Create the declared architecture-default shape/reference checkpoint."""

    output = Path(output_path).resolve()
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata != {
            "kind": "ood_architecture_default_template",
            "num_classes": int(num_classes),
            "width_mult": float(width_mult),
            "dropout": 0.15,
            "seed": int(seed),
            "contains_trained_target_weights": False,
        }:
            raise FileExistsError(f"existing OOD template contract differs: {output}")
        return ArtifactRef.create(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ResNet18Slim(o_dim=int(num_classes), width_mult=float(width_mult), dropout=0.15, init_type="kaiming_uniform")
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": {
                "kind": "ood_architecture_default_template",
                "num_classes": int(num_classes),
                "width_mult": float(width_mult),
                "dropout": 0.15,
                "seed": int(seed),
                "contains_trained_target_weights": False,
            },
        },
        output,
    )
    return ArtifactRef.create(output)


def _ours_ood_geometry(model: ResNet18Slim) -> tuple[list[LayerTileLayout], torch.Tensor]:
    layouts: list[LayerTileLayout] = []
    offset = 0
    for spec in supported_operator_specs(model, include_head=False):
        rows, cols = spec.matrix_shape
        row_tiles, col_tiles = (rows + 127) // 128, (cols + 127) // 128
        indices = tuple(range(offset, offset + row_tiles * col_tiles))
        layouts.append(LayerTileLayout(spec.module_name, indices, spec.matrix_shape, (128, 128)))
        offset += len(indices)
    # The OOD conditioning bundle is codec-neutral. Store a one-token semantic
    # template; runtime retokenizes it only after a sealed ours flow fixes T.
    return layouts, _ours_architecture_features(layouts, model.state_dict(), token_count=1)


def build_ood_conditioning_bundle(
    *,
    dataset_id: str,
    dataset_ref: ArtifactRef,
    template_ref: ArtifactRef,
    dataset_encoder: Any,
    dataset_encoder_provenance: Mapping[str, Any],
    output_path: str | Path,
    context_indices: Sequence[int],
    prompt_indices: Sequence[int],
    prompt_candidate_indices: Sequence[Sequence[int]] | None = None,
    weightclip_decoder: OfficialWeightCLIPMultiWindowDecoderAdapter | None = None,
) -> dict[str, Any]:
    dataset = load_train_dataset_pt(dataset_ref)
    template_payload = torch.load(template_ref.verify("OOD architecture template"), map_location="cpu", weights_only=False)
    metadata = template_payload.get("metadata", {})
    if metadata.get("kind") != "ood_architecture_default_template" or metadata.get("contains_trained_target_weights") is not False:
        raise ValueError("OOD template is not declared architecture-default/non-trained")
    state = load_checkpoint_state(template_ref)
    model = ResNet18Slim(
        o_dim=int(state["fc.weight"].shape[0]),
        width_mult=float(state["conv1.weight"].shape[0]) / 64.0,
        dropout=0.15,
        init_type=None,
    )
    model.load_state_dict(state, strict=True)
    layouts, ours_features = _ours_ood_geometry(model)
    context_images, context_provenance = _context_contract(
        dataset["trainset"].data,
        indices=context_indices,
        pool_id=f"{dataset_id}:train-context-v1",
        dataset_manifest_sha256=dataset_ref.sha256,
    )
    embedding = encode_dataset_embedding(dataset_encoder, dataset["trainset"].data, prompt_indices)
    embedding_candidates = None
    if prompt_candidate_indices is not None:
        if not prompt_candidate_indices:
            raise ValueError("prompt candidate list cannot be empty")
        embedding_candidates = torch.stack(
            [encode_dataset_embedding(dataset_encoder, dataset["trainset"].data, indices) for indices in prompt_candidate_indices]
        )
    weightclip_template_z_enc = (
        None if weightclip_decoder is None else weightclip_decoder.encode_anchor().detach().cpu().float().contiguous()
    )
    weightclip_template_mask = (
        None if weightclip_decoder is None else weightclip_decoder.window_token_mask.detach().cpu().bool().contiguous()
    )
    payload = {
        "schema_version": 1,
        "bundle_kind": "ood_conditioning",
        "identity": {"dataset_id": str(dataset_id), "split": "ood", "contains_target_weights": False},
        "dataset_pt": asdict(dataset_ref),
        "template_checkpoint": asdict(template_ref),
        "num_classes": int(state["fc.weight"].shape[0]),
        "width_mult": float(state["conv1.weight"].shape[0]) / 64.0,
        "dataset_embedding": embedding,
        "dataset_embedding_candidates": embedding_candidates,
        "dataset_prompt_provenance": {
            "source_split": "train",
            "image_indices": [int(index) for index in prompt_indices],
            "dataset_pt_sha256": dataset_ref.sha256,
            "candidate_image_indices": (
                None
                if prompt_candidate_indices is None
                else [[int(index) for index in indices] for indices in prompt_candidate_indices]
            ),
        },
        "context_images": context_images,
        "context_provenance": context_provenance,
        "ours_layouts": [asdict(layout) for layout in layouts],
        "ours_architecture_features": ours_features,
        "weightclip_architecture_features": (
            None if weightclip_decoder is None else weightclip_window_architecture_features(weightclip_decoder)
        ),
        "weightclip_template_z_enc": weightclip_template_z_enc,
        "weightclip_template_token_mask": weightclip_template_mask,
        "weightclip_decoder_provenance": None if weightclip_decoder is None else weightclip_decoder.provenance,
        "weightclip_codec_fingerprint": (
            None if weightclip_decoder is None else canonical_codec_fingerprint("weightclip", weightclip_decoder.provenance)
        ),
        "provenance": {
            "dataset_encoder": dict(dataset_encoder_provenance),
            "target_weight_access": "forbidden_and_absent",
            "source_state_policy": "architecture_default_template_only",
            "weightclip_template_code_sha256": (
                None if weightclip_template_z_enc is None else context_tensor_sha256(weightclip_template_z_enc)
            ),
        },
    }
    return _save_bundle(Path(output_path), payload)


def _context_contract(
    train_images: torch.Tensor,
    *,
    indices: Sequence[int],
    pool_id: str,
    dataset_manifest_sha256: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not indices or len(indices) > 512 or len(set(int(index) for index in indices)) != len(indices):
        raise ValueError("activation context requires 1..512 distinct train indices")
    if min(indices) < 0 or max(indices) >= len(train_images):
        raise IndexError("activation context index is outside the train split")
    images = train_images[list(indices)].detach().cpu().float().contiguous()
    provenance = {
        "source_split": "train",
        "manifest_recorded": True,
        "pool_id": str(pool_id),
        "manifest_sha256": str(dataset_manifest_sha256),
        "image_indices": [int(index) for index in indices],
        "images_sha256": context_tensor_sha256(images),
    }
    return images, provenance


def _identity(checkpoint_row: Mapping[str, Any], *, codec: str) -> dict[str, str]:
    checkpoint_id = str(checkpoint_row["checkpoint_sha256"])
    return {
        "group_id": f"{checkpoint_row['dataset']}:{checkpoint_row['lineage_id']}:{checkpoint_id[:12]}",
        "dataset_id": str(checkpoint_row["dataset"]),
        "lineage_id": str(checkpoint_row["lineage_id"]),
        "checkpoint_id": checkpoint_id,
        "split": str(checkpoint_row["split"]),
        "codec": str(codec),
    }


def _save_bundle(output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temp)
    proposed = ArtifactRef(str(output.resolve()), sha256_file(temp), temp.stat().st_size)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_payload = {
        "schema_version": 1,
        "kind": payload["bundle_kind"],
        "identity": payload["identity"],
        "bundle": asdict(proposed),
        "provenance": payload["provenance"],
    }
    if output.exists() or manifest_path.exists():
        if not output.is_file() or not manifest_path.is_file():
            temp.unlink(missing_ok=True)
            raise FileExistsError(f"partial immutable bundle commit exists: {output}")
        current = ArtifactRef.create(output)
        if current.sha256 != proposed.sha256 or current.bytes != proposed.bytes:
            temp.unlink(missing_ok=True)
            raise FileExistsError(f"immutable bundle already exists with different bytes: {output}")
        # Validate manifest bytes before touching the committed PT.
        write_json_immutable(manifest_path, manifest_payload)
        temp.unlink(missing_ok=True)
        return {"bundle": asdict(current), "manifest": asdict(ArtifactRef.create(manifest_path))}
    temp.replace(output)
    reference = ArtifactRef.create(output)
    write_json_immutable(
        manifest_path,
        manifest_payload,
    )
    return {"bundle": asdict(reference), "manifest": asdict(ArtifactRef.create(manifest_path))}


def build_ours_task_fit_bundle(
    *,
    checkpoint_row: Mapping[str, Any],
    dataset_ref: ArtifactRef,
    pair_manifest_ref: ArtifactRef | None,
    codec_seal_ref: ArtifactRef,
    decoder: BigVAETileDecoderAdapter,
    dataset_encoder: Any,
    dataset_encoder_provenance: Mapping[str, Any],
    output_path: str | Path,
    context_indices: Sequence[int],
    prompt_indices: Sequence[int],
    prompt_candidate_count: int = 100,
    prompt_candidate_seed: int = 0,
    encode_batch_size: int = 8,
    operator_samples: Sequence[OperatorBankSample] | None = None,
    train_only_dataset: bool = False,
) -> dict[str, Any]:
    if decoder.provenance.get("codec") != "ours":
        raise ValueError("ours bundle requires an ours decoder")
    codec_seal = json.loads(codec_seal_ref.verify("ours codec seal").read_text())
    sealed_checkpoint = codec_seal.get("checkpoint", {})
    if codec_seal.get("codec") != "ours" or sealed_checkpoint.get("sha256") != decoder.provenance.get("checkpoint_sha256"):
        raise ValueError("ours decoder disagrees with sealed codec provenance")
    if tuple(codec_seal.get("tile_shape", [])) != (128, 128) or not bool(codec_seal.get("deterministic_ae", False)):
        raise ValueError("ours codec seal must declare deterministic 128x128 AE")
    dataset = load_train_dataset_pt(dataset_ref) if train_only_dataset else load_dataset_pt(dataset_ref)
    source_ref = ArtifactRef.create(checkpoint_row["checkpoint_path"])
    if source_ref.sha256 != checkpoint_row["checkpoint_sha256"]:
        raise ValueError("checkpoint manifest SHA disagrees with source checkpoint")
    source_state = load_checkpoint_state(source_ref)
    checkpoint_id = str(checkpoint_row["checkpoint_sha256"])
    samples = list(operator_samples or ())
    if not samples:
        if pair_manifest_ref is None:
            raise ValueError("ours bundle requires either a matching operator bank or explicit operator samples")
        pair_path = pair_manifest_ref.verify("operator pair manifest")
        bank = OperatorBankTrainingDataset(
            pair_path,
            seed=0,
            repeat=False,
            permutation_views=False,
            canonical_probability=1.0,
            expected_pair_manifest_sha256=pair_manifest_ref.sha256,
        )
        for key in sorted(bank.operator_groups):
            if key[0] == checkpoint_id:
                samples.extend(bank._materialize_group(key, None))
        if not samples:
            raise ValueError(f"operator bank has no canonical records for checkpoint {checkpoint_id}")
    samples.sort(key=lambda item: (int(item.meta["operator"]["depth_index"]), item.layer_name, item.meta["tile_row"], item.meta["tile_col"]))
    device = next(decoder.model.parameters()).device
    code_chunks: list[torch.Tensor] = []
    for start in range(0, len(samples), int(encode_batch_size)):
        batch = samples[start : start + int(encode_batch_size)]
        W = torch.stack([item.weight for item in batch]).to(device)
        X = torch.stack([item.x for item in batch]).to(device)
        x_mask = torch.stack([item.meta["x_mask"] for item in batch]).to(device)
        d_in_mask = torch.stack([item.meta["d_in_mask"] for item in batch]).to(device)
        d_out_mask = torch.stack([item.meta["d_out_mask"] for item in batch]).to(device)
        with torch.inference_mode():
            _decoded, codes, _logvar, _directions = decoder.model(
                W,
                X,
                x_mask=x_mask,
                d_in_mask=d_in_mask,
                d_out_mask=d_out_mask,
            )
        latent_cfg = decoder.model.cfg.big_vae
        expected_values = int(latent_cfg.num_latents) * int(latent_cfg.d_lat)
        if codes.ndim == 2 and codes.shape[1] == expected_values:
            codes = codes.reshape(len(batch), int(latent_cfg.num_latents), int(latent_cfg.d_lat))
        if codes.ndim != 3 or tuple(codes.shape[1:]) != (int(latent_cfg.num_latents), int(latent_cfg.d_lat)):
            raise ValueError(f"ours encoder emitted invalid grouped latent shape {tuple(codes.shape)}")
        code_chunks.append(codes.detach().cpu().float())
    z_enc = torch.cat(code_chunks)
    identities_parent = _identity(checkpoint_row, codec="ours")
    tile_identities = tuple(
        TileIdentity(
            identities_parent["dataset_id"],
            identities_parent["lineage_id"],
            identities_parent["checkpoint_id"],
            sample.layer_name.removesuffix(".weight"),
            int(sample.meta["tile_row"]),
            int(sample.meta["tile_col"]),
        )
        for sample in samples
    )
    layouts: list[LayerTileLayout] = []
    for layer_key in dict.fromkeys(sample.layer_name for sample in samples):
        indices = tuple(index for index, sample in enumerate(samples) if sample.layer_name == layer_key)
        operator = samples[indices[0]].meta["operator"]
        layouts.append(LayerTileLayout(layer_key.removesuffix(".weight"), indices, tuple(operator["matrix_shape"]), (128, 128)))
    context_images, context_provenance = _context_contract(
        dataset["trainset"].data,
        indices=context_indices,
        pool_id=f"{identities_parent['dataset_id']}:train-context-v1",
        dataset_manifest_sha256=dataset_ref.sha256,
    )
    embedding_bank, embedding_bank_provenance = encode_dataset_embedding_bank(
        dataset_encoder,
        dataset["trainset"].data,
        prompt_indices,
        candidate_count=prompt_candidate_count,
        seed=prompt_candidate_seed,
    )
    embedding_bank_provenance["dataset_encoder"] = dict(dataset_encoder_provenance)
    embedding = embedding_bank[0]
    payload = {
        "schema_version": 1,
        "bundle_kind": "ours_task_fit",
        "identity": identities_parent,
        "source_checkpoint": asdict(source_ref),
        "dataset_pt": asdict(dataset_ref),
        "z_enc": z_enc,
        "layouts": [asdict(layout) for layout in layouts],
        "tile_identities": [asdict(item) for item in tile_identities],
        "architecture_features": _ours_architecture_features(layouts, source_state, token_count=z_enc.shape[1]),
        "tile_mask": torch.ones(z_enc.shape[:2], dtype=torch.bool),
        "dataset_embedding": embedding,
        "dataset_embedding_bank": embedding_bank,
        "dataset_embedding_bank_provenance": embedding_bank_provenance,
        "dataset_prompt_provenance": {
            "source_split": "train",
            "image_indices": [int(index) for index in prompt_indices],
            "dataset_pt_sha256": dataset_ref.sha256,
        },
        "context_images": context_images,
        "context_provenance": context_provenance,
        "provenance": {
            "codec_seal": asdict(codec_seal_ref),
            "decoder": dict(decoder.provenance),
            "codec_fingerprint": canonical_codec_fingerprint("ours", decoder.provenance),
            "operator_pair_manifest": None if pair_manifest_ref is None else asdict(pair_manifest_ref),
            "operator_source": "train_only_on_the_fly" if pair_manifest_ref is None else "content_addressed_source_bank",
            "dataset_encoder": dict(dataset_encoder_provenance),
            "source_state_policy": "path_and_sha_only_no_tensor_duplication",
            "head_policy": "original_frozen_source",
            "bn_policy": "original_frozen_source",
            "architecture_conditioning": {
                "schema": list(FLOW_ARCHITECTURE_FEATURE_SCHEMA),
                "granularity": "128x128_operator_tile_repeated_over_ae_slots",
                "operator_matrix_orientation": "[in_times_kernel,out]",
                "codec_native_position_fields_excluded": True,
            },
        },
    }
    return _save_bundle(Path(output_path), payload)


def capture_train_only_operator_samples(
    *,
    checkpoint_ref: ArtifactRef,
    dataset_ref: ArtifactRef,
    context_indices: Sequence[int],
    device: str,
    batch_size: int = 32,
) -> list[OperatorBankSample]:
    """Build canonical operator tiles for one anchor without touching val/test."""

    dataset = load_train_dataset_pt(dataset_ref)
    state = load_checkpoint_state(checkpoint_ref)
    model = ResNet18Slim(
        o_dim=int(state["fc.weight"].shape[0]),
        width_mult=float(state["conv1.weight"].shape[0]) / 64.0,
        dropout=0.15,
        init_type=None,
    ).to(device)
    model.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)
    specs = supported_operator_specs(model, include_head=False)
    images = dataset["trainset"].data[list(context_indices)]
    labels = dataset["trainset"].targets[list(context_indices)]
    batches = [
        (images[start : start + batch_size], labels[start : start + batch_size])
        for start in range(0, len(images), batch_size)
    ]
    activations = capture_native_train_activations(
        model,
        batches,
        specs,
        max_rows=512,
        seed_parts=(checkpoint_ref.sha256, dataset_ref.sha256, "ood-anchor-train-only"),
        device=device,
    )
    output: list[OperatorBankSample] = []
    for spec in specs:
        contexts = {item.row_start: item for item in context_tiles(activations[spec.key], max_rows=512)}
        matrix = parameter_to_matrix(state[spec.key], spec).cpu().float()
        for tile, _mask, tile_spec in tile_matrix(matrix, 128, 128):
            context = contexts[tile_spec.row_start]
            d_in_mask = torch.zeros(128, dtype=torch.bool)
            d_in_mask[: tile_spec.valid_rows] = True
            d_out_mask = torch.zeros(128, dtype=torch.bool)
            d_out_mask[: tile_spec.valid_cols] = True
            output.append(
                OperatorBankSample(
                    x=context.raw_rows,
                    weight=tile,
                    meta={
                        "x_mask": context.sample_mask,
                        "d_in_mask": d_in_mask,
                        "d_out_mask": d_out_mask,
                        "operator": spec.to_dict(),
                        "tile_row": tile_spec.row_start // 128,
                        "tile_col": tile_spec.col_start // 128,
                        "activation_seed": stable_seed(checkpoint_ref.sha256, spec.key),
                        "native_activation_source": "dataset.trainset",
                    },
                    model_name=checkpoint_ref.sha256,
                    layer_name=spec.key,
                )
            )
    return output


def build_weightclip_task_fit_bundle(
    *,
    checkpoint_row: Mapping[str, Any],
    dataset_ref: ArtifactRef,
    decoder: OfficialWeightCLIPMultiWindowDecoderAdapter,
    dataset_encoder: Any,
    dataset_encoder_provenance: Mapping[str, Any],
    output_path: str | Path,
    context_indices: Sequence[int],
    prompt_indices: Sequence[int],
    prompt_candidate_count: int = 100,
    prompt_candidate_seed: int = 0,
    train_only_dataset: bool = False,
    template_checkpoint_ref: ArtifactRef | None = None,
    template_z_enc: torch.Tensor | None = None,
    template_token_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if decoder.provenance.get("codec") != "weightclip":
        raise ValueError("WeightCLIP bundle requires the official codec adapter")
    dataset = load_train_dataset_pt(dataset_ref) if train_only_dataset else load_dataset_pt(dataset_ref)
    source_ref = ArtifactRef.create(checkpoint_row["checkpoint_path"])
    if source_ref.sha256 != checkpoint_row["checkpoint_sha256"]:
        raise ValueError("checkpoint manifest SHA disagrees with source checkpoint")
    source_state = load_checkpoint_state(source_ref)
    if set(source_state) != set(decoder.anchor_state):
        raise ValueError("official WeightCLIP anchor state keys disagree with source checkpoint")
    for key in source_state:
        if not torch.equal(source_state[key], decoder.anchor_state[key].cpu().float()):
            raise ValueError(f"official WeightCLIP anchor tensor mismatch: {key}")
    with torch.inference_mode():
        z_enc = decoder.encode_anchor().detach().cpu().float()
    if any(value is not None for value in (template_checkpoint_ref, template_z_enc, template_token_mask)):
        if template_checkpoint_ref is None or template_z_enc is None or template_token_mask is None:
            raise ValueError("WeightCLIP default template checkpoint/code/mask must be provided together")
        template_payload = torch.load(
            template_checkpoint_ref.verify("WeightCLIP architecture-default template"),
            map_location="cpu",
            weights_only=False,
        )
        if template_payload.get("metadata", {}).get("contains_trained_target_weights") is not False:
            raise ValueError("WeightCLIP template checkpoint may contain trained target weights")
        template_z_enc = template_z_enc.detach().cpu().float().contiguous()
        template_token_mask = template_token_mask.detach().cpu().bool().contiguous()
        if tuple(template_z_enc.shape) != tuple(z_enc.shape):
            raise ValueError("WeightCLIP template/source code geometry mismatch")
        if tuple(template_token_mask.shape) != tuple(template_z_enc.shape[:2]):
            raise ValueError("WeightCLIP template mask/code geometry mismatch")
    identity = _identity(checkpoint_row, codec="weightclip")
    context_images, context_provenance = _context_contract(
        dataset["trainset"].data,
        indices=context_indices,
        pool_id=f"{identity['dataset_id']}:train-context-v1",
        dataset_manifest_sha256=dataset_ref.sha256,
    )
    embedding_bank, embedding_bank_provenance = encode_dataset_embedding_bank(
        dataset_encoder,
        dataset["trainset"].data,
        prompt_indices,
        candidate_count=prompt_candidate_count,
        seed=prompt_candidate_seed,
    )
    embedding_bank_provenance["dataset_encoder"] = dict(dataset_encoder_provenance)
    embedding = embedding_bank[0]
    payload = {
        "schema_version": 1,
        "bundle_kind": "weightclip_task_fit",
        "identity": identity,
        "source_checkpoint": asdict(source_ref),
        "dataset_pt": asdict(dataset_ref),
        "z_enc": z_enc,
        "weightclip_template_checkpoint": (
            None if template_checkpoint_ref is None else asdict(template_checkpoint_ref)
        ),
        "weightclip_template_z_enc": template_z_enc,
        "weightclip_template_token_mask": template_token_mask,
        "dataset_embedding": embedding,
        "dataset_embedding_bank": embedding_bank,
        "dataset_embedding_bank_provenance": embedding_bank_provenance,
        "dataset_prompt_provenance": {
            "source_split": "train",
            "image_indices": [int(index) for index in prompt_indices],
            "dataset_pt_sha256": dataset_ref.sha256,
        },
        "architecture_features": weightclip_window_architecture_features(decoder),
        "tile_mask": decoder.window_token_mask.detach().cpu(),
        "context_images": context_images,
        "context_provenance": context_provenance,
        "provenance": {
            "decoder": decoder.provenance,
            "codec_fingerprint": canonical_codec_fingerprint("weightclip", decoder.provenance),
            "dataset_encoder": dict(dataset_encoder_provenance),
            "windowing": "official_multiwindow_no_128x128_retiling",
            "head_policy": "original_frozen_source",
            "bn_policy": "original_frozen_source",
            "weightclip_template_code_sha256": (
                None if template_z_enc is None else context_tensor_sha256(template_z_enc)
            ),
            "architecture_conditioning": {
                "schema": list(FLOW_ARCHITECTURE_FEATURE_SCHEMA),
                "granularity": "official_sparse_token_mapped_to_first_scalar_conceptual_128x128_tile",
                "operator_matrix_orientation": "[in_times_kernel,out]",
                "known_deviation": "one_official_sparse_token_may_span_multiple_conceptual_input_tiles",
                "codec_native_position_fields_excluded": True,
            },
        },
    }
    return _save_bundle(Path(output_path), payload)


def load_task_fit_bundle(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported task-fit bundle schema")
    if expected_kind is not None and payload.get("bundle_kind") != expected_kind:
        raise ValueError(f"task-fit bundle kind mismatch: {payload.get('bundle_kind')} != {expected_kind}")
    ArtifactRef.from_mapping(payload["source_checkpoint"]).verify("bundle source checkpoint")
    dataset_ref = ArtifactRef.from_mapping(payload["dataset_pt"])
    if str(payload.get("identity", {}).get("split", "")).startswith("ood_"):
        load_train_dataset_pt(dataset_ref)
    else:
        load_dataset_pt(dataset_ref)
    provenance = payload.get("provenance", {})
    if payload.get("bundle_kind") == "ours_task_fit":
        ArtifactRef.from_mapping(provenance["codec_seal"]).verify("bundle ours codec seal")
        pair_manifest = provenance.get("operator_pair_manifest")
        if pair_manifest is None:
            if provenance.get("operator_source") != "train_only_on_the_fly":
                raise ValueError("ours bundle omits operator bank without a train-only extraction declaration")
        else:
            ArtifactRef.from_mapping(pair_manifest).verify("bundle operator pair manifest")
        if provenance.get("source_state_policy") != "path_and_sha_only_no_tensor_duplication":
            raise ValueError("ours bundle source-state policy drift")
        if provenance.get("decoder", {}).get("codec") != "ours":
            raise ValueError("ours bundle decoder provenance is missing")
        if provenance.get("codec_fingerprint") != canonical_codec_fingerprint("ours", provenance["decoder"]):
            raise ValueError("ours bundle decoder fingerprint mismatch")
        if "layouts" not in payload or "tile_identities" not in payload:
            raise ValueError("ours bundle is missing topological tile layout")
    elif payload.get("bundle_kind") == "weightclip_task_fit":
        decoder_provenance = provenance.get("decoder", {})
        if decoder_provenance.get("codec") != "weightclip":
            raise ValueError("WeightCLIP bundle decoder provenance is missing")
        if provenance.get("codec_fingerprint") != canonical_codec_fingerprint("weightclip", decoder_provenance):
            raise ValueError("WeightCLIP bundle decoder fingerprint mismatch")
        template_code = payload.get("weightclip_template_z_enc")
        template_mask = payload.get("weightclip_template_token_mask")
        template_ref_raw = payload.get("weightclip_template_checkpoint")
        if any(value is not None for value in (template_code, template_mask, template_ref_raw)):
            if not torch.is_tensor(template_code) or not torch.is_tensor(template_mask) or template_ref_raw is None:
                raise ValueError("WeightCLIP bundle has a partial architecture-default template")
            template_ref = ArtifactRef.from_mapping(template_ref_raw)
            template_payload = torch.load(
                template_ref.verify("WeightCLIP bundle architecture-default template"),
                map_location="cpu",
                weights_only=False,
            )
            if template_payload.get("metadata", {}).get("contains_trained_target_weights") is not False:
                raise ValueError("WeightCLIP bundle template may contain trained target weights")
            if tuple(template_code.shape) != tuple(payload["z_enc"].shape):
                raise ValueError("WeightCLIP bundle template/source code geometry mismatch")
            if tuple(template_mask.shape) != tuple(template_code.shape[:2]):
                raise ValueError("WeightCLIP bundle template mask geometry mismatch")
            if context_tensor_sha256(template_code) != provenance.get("weightclip_template_code_sha256"):
                raise ValueError("WeightCLIP bundle template code hash mismatch")
    else:
        raise ValueError(f"unsupported task-fit bundle kind {payload.get('bundle_kind')!r}")
    if payload["context_provenance"].get("source_split") != "train":
        raise ValueError("task-fit context pool is not train-only")
    if context_tensor_sha256(payload["context_images"]) != payload["context_provenance"].get("images_sha256"):
        raise ValueError("task-fit context pool tensor hash mismatch")
    z_enc = payload.get("z_enc")
    if not torch.is_tensor(z_enc) or z_enc.ndim != 3 or not torch.isfinite(z_enc).all():
        raise ValueError("task-fit bundle has invalid grouped z_enc")
    if tuple(payload["tile_mask"].shape) != tuple(z_enc.shape[:2]):
        raise ValueError("task-fit bundle latent mask disagrees with z_enc")
    return payload


def load_ood_conditioning_bundle(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 1 or payload.get("bundle_kind") != "ood_conditioning":
        raise ValueError("not an OOD conditioning bundle v1")
    identity = payload.get("identity", {})
    if identity.get("contains_target_weights") is not False or identity.get("split") != "ood":
        raise ValueError("OOD conditioning bundle target-weight guard is absent")
    load_train_dataset_pt(ArtifactRef.from_mapping(payload["dataset_pt"]))
    template_ref = ArtifactRef.from_mapping(payload["template_checkpoint"])
    template = torch.load(template_ref.verify("OOD template"), map_location="cpu", weights_only=False)
    if template.get("metadata", {}).get("contains_trained_target_weights") is not False:
        raise ValueError("OOD conditioning template may contain trained target weights")
    if payload.get("provenance", {}).get("target_weight_access") != "forbidden_and_absent":
        raise ValueError("OOD conditioning provenance permits target-weight access")
    if payload["context_provenance"].get("source_split") != "train":
        raise ValueError("OOD conditioning context is not train-only")
    if context_tensor_sha256(payload["context_images"]) != payload["context_provenance"].get("images_sha256"):
        raise ValueError("OOD conditioning context hash mismatch")
    if "z_enc" in payload or "source_checkpoint" in payload:
        raise ValueError("OOD conditioning bundle illegally contains target weights/anchors")
    wc_provenance = payload.get("weightclip_decoder_provenance")
    if wc_provenance is not None and payload.get("weightclip_codec_fingerprint") != canonical_codec_fingerprint(
        "weightclip", wc_provenance
    ):
        raise ValueError("OOD conditioning WeightCLIP fingerprint mismatch")
    wc_template = payload.get("weightclip_template_z_enc")
    wc_mask = payload.get("weightclip_template_token_mask")
    if wc_provenance is not None:
        if not torch.is_tensor(wc_template) or wc_template.ndim != 3 or not torch.isfinite(wc_template).all():
            raise ValueError("OOD conditioning WeightCLIP template code is invalid")
        if not torch.is_tensor(wc_mask) or tuple(wc_mask.shape) != tuple(wc_template.shape[:2]):
            raise ValueError("OOD conditioning WeightCLIP template mask disagrees with code")
        expected_hash = payload.get("provenance", {}).get("weightclip_template_code_sha256")
        if context_tensor_sha256(wc_template) != expected_hash:
            raise ValueError("OOD conditioning WeightCLIP template code hash mismatch")
    return payload
