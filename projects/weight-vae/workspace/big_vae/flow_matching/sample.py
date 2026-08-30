from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from big_vae.weightclip_benchmark.resnet_functional import ConvSpec, ConvWeightProvider, FunctionalResNet18Slim, FunctionalResNetConfig
from big_vae.weightclip_benchmark.resnet18slim import ResNet18Slim
from big_vae.weightclip_benchmark.task_latent_fit import (
    LayerTileLayout,
    TileDecoder,
    context_tensor_sha256,
    validate_context_provenance,
)

from .dataset import LatentNormalizer, canonical_codec_fingerprint
from .solvers import IntegrationResult, integrate_flow


@dataclass(slots=True)
class FlowSamplingConfig:
    path_kind: str
    solver: str = "heun"
    nfe_steps: int = 16
    seed: int = 0
    activation_conditioning: bool = True


@dataclass(slots=True)
class LoadedSealedFlow:
    model: nn.Module
    normalizer: LatentNormalizer
    seal: dict[str, object]
    checkpoint_path: Path
    used_ema: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(checkpoint: Mapping[str, object]) -> str:
    payload = {"model_config": checkpoint["model_config"], "train_config": checkpoint["train_config"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal_flow_checkpoint(
    *,
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
    seal_path: str | Path,
    codec: str,
    decoded_validation_path: str | Path,
    global_selection_path: str | Path,
) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    normalizer_path = Path(normalizer_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("schema_version", -1)) != 3:
        raise ValueError("flow checkpoint predates immutable resume/validation contract v3")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("codec") != codec:
        raise ValueError(f"checkpoint codec {provenance.get('codec')!r} != requested {codec!r}")
    codec_fingerprint = provenance.get("codec_fingerprint")
    if not isinstance(codec_fingerprint, str) or not codec_fingerprint.startswith("sha256:") or len(codec_fingerprint) != 71:
        raise ValueError("checkpoint lacks canonical grouped-record codec_fingerprint")
    resume_contract = checkpoint.get("resume_contract_sha256")
    if not isinstance(resume_contract, str) or len(resume_contract) != 64:
        raise ValueError("checkpoint lacks immutable resume contract SHA-256")
    validation_metrics = checkpoint.get("validation_metrics")
    if not isinstance(validation_metrics, Mapping):
        raise ValueError("checkpoint lacks EMA endpoint validation metrics")
    if float(validation_metrics.get("ema_endpoint_finite_fraction", -1.0)) != 1.0:
        raise ValueError("refusing to seal a flow with non-finite EMA endpoints")
    normalizer = LatentNormalizer.from_state_dict(torch.load(normalizer_path, map_location="cpu", weights_only=False))
    if normalizer.fit_split != "train":
        raise ValueError("refusing to seal a normalizer not fit on train")
    if normalizer.codec_fingerprint != codec_fingerprint:
        raise ValueError("normalizer codec fingerprint disagrees with flow checkpoint")
    decoded_validation_path = Path(decoded_validation_path).resolve()
    decoded_report = json.loads(decoded_validation_path.read_text())
    if int(decoded_report.get("schema_version", -1)) != 1:
        raise ValueError("unsupported decoded flow validation report schema")
    if decoded_report.get("kind") != "flow_decoded_e4_sweep":
        raise ValueError("decoded validation report is not a frozen E4 sweep artifact")
    expected_report_contract = {
        "codec": codec,
        "codec_fingerprint": codec_fingerprint,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "normalizer_sha256": _sha256_file(normalizer_path),
    }
    for key, expected in expected_report_contract.items():
        if decoded_report.get(key) != expected:
            raise ValueError(f"decoded validation report {key} disagrees with flow substrate")
    global_selection_path = Path(global_selection_path).resolve()
    global_report = json.loads(global_selection_path.read_text())
    if int(global_report.get("schema_version", -1)) != 1 or global_report.get("kind") != "flow_global_e4_selection":
        raise ValueError("global E4 artifact has an unsupported contract")
    selection = global_report.get("selection")
    if not isinstance(selection, Mapping) or selection.get("solver") not in {"euler", "heun"}:
        raise ValueError("global E4 report lacks a valid common solver selection")
    if int(selection.get("nfe_steps", -1)) not in {4, 8, 16, 32}:
        raise ValueError("decoded validation report selected an out-of-grid NFE")
    sweep = decoded_report.get("sweep")
    if not isinstance(sweep, list) or {
        (str(row.get("solver")), int(row.get("nfe_steps", -1))) for row in sweep if isinstance(row, Mapping)
    } != {(solver, steps) for solver in ("euler", "heun") for steps in (4, 8, 16, 32)}:
        raise ValueError("decoded validation report does not contain the complete frozen E4 sweep")
    expected_times = {"0.0", "0.25", "0.5", "0.75", "1.0"}
    if any(set(row.get("decoded_health_by_time", {})) != expected_times for row in sweep):
        raise ValueError("decoded validation report lacks frozen E4 trajectory health times")
    report_refs = global_report.get("reports")
    if not isinstance(report_refs, list):
        raise ValueError("global E4 report lacks provisional report artifact references")
    observed_arms = {
        (str(row.get("codec")), str(row.get("path_kind"))) for row in report_refs if isinstance(row, Mapping)
    }
    expected_arms = {
        ("ours", "gaussian"),
        ("ours", "paired_anchor"),
        ("weightclip", "gaussian"),
        ("weightclip", "paired_anchor"),
    }
    if len(report_refs) != 4 or observed_arms != expected_arms:
        raise ValueError("global E4 selection does not bind exactly the four matched flow arms")
    decoded_sha = _sha256_file(decoded_validation_path)
    matching_refs = [
        row
        for row in report_refs
        if isinstance(row, Mapping)
        and row.get("sha256") == decoded_sha
        and row.get("codec") == codec
        and row.get("path_kind") == checkpoint["train_config"]["path_kind"]
    ]
    if len(matching_refs) != 1:
        raise ValueError("global E4 selection does not bind this exact arm report")
    selected_rows = [
        row for row in sweep if row["solver"] == selection["solver"] and int(row["nfe_steps"]) == int(selection["nfe_steps"])
    ]
    if len(selected_rows) != 1:
        raise ValueError("common global E4 selection is absent from arm sweep")
    selected_row = selected_rows[0]
    if float(selected_row.get("decoded_finite_fraction", -1.0)) != 1.0:
        raise ValueError("common E4 selection has non-finite decoded weights for this arm")
    role_health = selected_row["decoded_health_by_time"]["1.0"]
    required_roles = {"stem", "residual_conv1", "residual_conv2", "projection"}
    if not isinstance(role_health, Mapping) or not required_roles <= set(role_health):
        raise ValueError("selected decoded validation row lacks required body-role health")
    for role in required_roles:
        row = role_health[role]
        if not isinstance(row, Mapping) or int(row.get("count", 0)) <= 0:
            raise ValueError(f"decoded validation role {role!r} is empty")
        if not torch.isfinite(torch.tensor(float(row.get("weight_rms", float("nan"))))):
            raise ValueError(f"decoded validation role {role!r} has non-finite weight health")
    seal = {
        "schema_version": 4,
        "codec": codec,
        "codec_fingerprint": codec_fingerprint,
        "path_kind": checkpoint["train_config"]["path_kind"],
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "normalizer_sha256": _sha256_file(normalizer_path),
        "config_fingerprint_sha256": _config_fingerprint(checkpoint),
        "checkpoint_name": checkpoint_path.name,
        "normalizer_name": normalizer_path.name,
        "step": int(checkpoint["step"]),
        "ema_available": "ema" in checkpoint and "shadow" in checkpoint["ema"],
        "activation_conditioning": provenance.get("activation_conditioning"),
        "solver": selection["solver"],
        "nfe_steps": int(selection["nfe_steps"]),
        "actual_nfe": int(selection["actual_nfe"]),
        "selection_metric": "common_four_arm_mean_pooled_endpoint_rmse_with_decoded_health_gate",
        "resume_contract_sha256": resume_contract,
        "validation_metrics": dict(validation_metrics),
        "decoded_validation_artifact": {
            "path": str(decoded_validation_path),
            "sha256": decoded_sha,
            "bytes": decoded_validation_path.stat().st_size,
        },
        "global_e4_selection_artifact": {
            "path": str(global_selection_path),
            "sha256": _sha256_file(global_selection_path),
            "bytes": global_selection_path.stat().st_size,
        },
        "decoded_validation_metrics": {
            "decoded_models": int(selected_row["decoded_models"]),
            "decoded_finite_fraction": float(selected_row["decoded_finite_fraction"]),
            "ema_endpoint_rmse": float(selected_row["endpoint_rmse"]),
            "role_health": role_health,
        },
    }
    output = Path(seal_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError(f"immutable flow seal conflict: {output}")
        return output
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    return output


def load_sealed_flow(
    *,
    seal_path: str | Path,
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
    expected_codec: str,
    expected_path_kind: str,
    device: str = "cpu",
    use_ema: bool = True,
) -> LoadedSealedFlow:
    from .model import ConditionalVelocityTransformer, FlowModelConfig

    seal = json.loads(Path(seal_path).read_text())
    if int(seal.get("schema_version", -1)) != 4:
        raise ValueError("unsupported flow seal schema")
    if seal.get("codec") != expected_codec or seal.get("path_kind") != expected_path_kind:
        raise ValueError(
            f"flow substrate mismatch: seal codec/path={seal.get('codec')}/{seal.get('path_kind')}, "
            f"expected={expected_codec}/{expected_path_kind}"
        )
    checkpoint_path = Path(checkpoint_path).resolve()
    normalizer_path = Path(normalizer_path).resolve()
    if _sha256_file(checkpoint_path) != seal["checkpoint_sha256"]:
        raise ValueError("sealed flow checkpoint SHA-256 mismatch")
    if _sha256_file(normalizer_path) != seal["normalizer_sha256"]:
        raise ValueError("sealed latent normalizer SHA-256 mismatch")
    for key in ("decoded_validation_artifact", "global_e4_selection_artifact"):
        artifact = seal.get(key)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"flow seal lacks {key}")
        artifact_path = Path(str(artifact.get("path", "")))
        if not artifact_path.is_file() or _sha256_file(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"sealed {key} is absent or its SHA-256 mismatches")
        if artifact_path.stat().st_size != int(artifact.get("bytes", -1)):
            raise ValueError(f"sealed {key} byte count mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("schema_version", -1)) != 3:
        raise ValueError("sealed flow checkpoint schema is not immutable-contract v3")
    if _config_fingerprint(checkpoint) != seal["config_fingerprint_sha256"]:
        raise ValueError("sealed flow config fingerprint mismatch")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("codec") != expected_codec:
        raise ValueError("checkpoint provenance codec disagrees with seal")
    if checkpoint["train_config"]["path_kind"] != expected_path_kind:
        raise ValueError("checkpoint train path disagrees with seal")
    if provenance.get("activation_conditioning") != seal.get("activation_conditioning"):
        raise ValueError("checkpoint activation-conditioning provenance disagrees with seal")
    if provenance.get("codec_fingerprint") != seal.get("codec_fingerprint"):
        raise ValueError("checkpoint codec fingerprint disagrees with flow seal")
    if checkpoint.get("resume_contract_sha256") != seal.get("resume_contract_sha256"):
        raise ValueError("checkpoint resume contract disagrees with flow seal")
    model = ConditionalVelocityTransformer(FlowModelConfig(**checkpoint["model_config"]))
    if use_ema:
        if not seal.get("ema_available"):
            raise ValueError("EMA requested but absent from sealed checkpoint")
        state = checkpoint["ema"]["shadow"]
    else:
        state = checkpoint["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    normalizer = LatentNormalizer.from_state_dict(torch.load(normalizer_path, map_location="cpu", weights_only=False))
    if normalizer.fit_split != "train":
        raise ValueError("sealed normalizer fit_split is not train")
    if normalizer.codec_fingerprint != seal.get("codec_fingerprint"):
        raise ValueError("sealed normalizer codec fingerprint disagrees with flow seal")
    return LoadedSealedFlow(model, normalizer, seal, checkpoint_path, use_ema)


def assert_decoder_matches_loaded_flow(loaded_flow: LoadedSealedFlow, decoder: object) -> str:
    provenance = getattr(decoder, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime decoder lacks immutable provenance for codec fingerprint verification")
    codec = str(loaded_flow.seal.get("codec", ""))
    actual = canonical_codec_fingerprint(codec, provenance)
    expected = loaded_flow.seal.get("codec_fingerprint")
    if actual != expected:
        raise ValueError(
            f"runtime decoder codec fingerprint mismatch: flow={expected!r} decoder={actual!r}"
        )
    return actual


class FlowDecodedConvProvider(ConvWeightProvider):
    """Generate a layer's codes, then decode it using its generated-prefix context."""

    def __init__(
        self,
        *,
        flow: nn.Module,
        decoder: TileDecoder,
        layouts: Sequence[LayerTileLayout],
        architecture_features: torch.Tensor,
        dataset_embedding: torch.Tensor,
        normalizer: LatentNormalizer,
        config: FlowSamplingConfig,
        anchors: torch.Tensor | None = None,
        tile_mask: torch.Tensor | None = None,
    ) -> None:
        self.flow = flow
        self.decoder = decoder
        self.layouts = {layout.layer_key: layout for layout in layouts}
        self.architecture_features = architecture_features
        self.dataset_embedding = dataset_embedding
        self.normalizer = normalizer
        self.config = config
        self.anchors = anchors
        self.tile_mask = tile_mask
        self.generated_codes: dict[str, torch.Tensor] = {}
        self.weights_by_layer: dict[str, torch.Tensor] = {}
        self.nfe_by_layer: dict[str, int] = {}
        if config.path_kind == "paired_anchor" and anchors is None:
            raise ValueError("paired_anchor sampling requires clean z_enc anchors")
        if config.path_kind == "gaussian" and anchors is not None:
            raise ValueError("anchor-free sampling must not receive anchors")

    def _base(self, layout: LayerTileLayout, indices: torch.Tensor) -> torch.Tensor:
        if self.config.path_kind == "paired_anchor":
            assert self.anchors is not None
            return self.normalizer.normalize(self.anchors.index_select(0, indices))
        latent_shape = self.anchors.shape[1:] if self.anchors is not None else None
        if latent_shape is None:
            latent_dim = int(getattr(self.flow, "config").latent_dim)
            # Tile codecs may have several latent tokens; derive it from token mask.
            token_count = int(self.tile_mask.shape[1]) if self.tile_mask is not None else 1
            latent_shape = (token_count, latent_dim)
        digest = hashlib.sha256(f"{self.config.seed}:{layout.layer_key}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=self.architecture_features.device).manual_seed(seed)
        return torch.randn((len(indices), *latent_shape), device=self.architecture_features.device, generator=generator)

    def _sample_codes(self, layout: LayerTileLayout) -> tuple[torch.Tensor, IntegrationResult]:
        indices = torch.tensor(layout.code_indices, device=self.architecture_features.device, dtype=torch.long)
        base = self._base(layout, indices)
        features = self.architecture_features.index_select(0, indices)
        dataset = self.dataset_embedding.unsqueeze(0).expand(len(indices), -1)
        mask = None if self.tile_mask is None else self.tile_mask.index_select(0, indices)

        def velocity(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.flow(z, t, dataset_embedding=dataset, architecture_features=features, token_mask=mask)

        result = integrate_flow(velocity, base, method=self.config.solver, steps=self.config.nfe_steps)
        return self.normalizer.denormalize(result.endpoint), result

    def get_conv_weight(self, spec: ConvSpec, activation_context: torch.Tensor) -> torch.Tensor:
        layout = self.layouts[spec.key]
        codes, integration = self._sample_codes(layout)
        self.generated_codes[spec.key] = codes.detach().cpu()
        self.nfe_by_layer[spec.key] = integration.nfe
        indices = torch.tensor(layout.code_indices, device=codes.device, dtype=torch.long)
        features = self.architecture_features.index_select(0, indices)
        mask = None if self.tile_mask is None else self.tile_mask.index_select(0, indices)
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
        tiles = self.decoder.decode_tiles(
            codes,
            activation_context=torch.stack(contexts) if self.config.activation_conditioning else None,
            architecture_features=features,
            tile_mask=mask,
            weight_mask=torch.stack(weight_masks),
            tile_indices=indices,
            layer_key=spec.key,
        )
        expected = (rows * cols, tile_h, tile_w)
        if tuple(tiles.shape) != expected:
            raise ValueError(f"decoder returned {tuple(tiles.shape)} for {spec.key}, expected {expected}")
        matrix = torch.cat([torch.cat([tiles[r * cols + c] for c in range(cols)], dim=1) for r in range(rows)], dim=0)
        matrix = matrix[: layout.matrix_shape[0], : layout.matrix_shape[1]]
        weight = matrix.transpose(0, 1).reshape(spec.weight_shape).contiguous()
        self.weights_by_layer[spec.key] = weight.detach().clone()
        return weight


def build_flow_materialized_resnet(
    *,
    model_config: FunctionalResNetConfig,
    provider: FlowDecodedConvProvider,
    source_state: Mapping[str, torch.Tensor] | None,
    downstream_head_seed: int,
) -> FunctionalResNet18Slim:
    """Construct topological sequential materializer with default downstream head/BN.

    The downstream contract is encoded here even when task fitting used the configurable
    original frozen critic: generated models always use a fresh paired random head and our
    method's default BN state.
    """
    downstream_config = FunctionalResNetConfig(**{**model_config.__dict__}) if hasattr(model_config, "__dict__") else FunctionalResNetConfig(
        width_mult=model_config.width_mult,
        channels_in=model_config.channels_in,
        num_classes=model_config.num_classes,
        activation=model_config.activation,
        context_rows=model_config.context_rows,
        context_seed=model_config.context_seed,
        dropout=model_config.dropout,
    )
    downstream_config.head_policy = "default_random_frozen"
    downstream_config.bn_policy = "default_frozen"
    return FunctionalResNet18Slim(downstream_config, provider, source_state=source_state, random_head_seed=downstream_head_seed)


def materialize_standard_resnet(
    *,
    provider: FlowDecodedConvProvider,
    model_config: FunctionalResNetConfig,
    activation_images: torch.Tensor,
    downstream_head_seed: int,
) -> ResNet18Slim:
    """Run sequential generation once and return an ordinary trainable ResNet."""
    provider.generated_codes.clear()
    provider.weights_by_layer.clear()
    functional = build_flow_materialized_resnet(
        model_config=model_config,
        provider=provider,
        source_state=None,
        downstream_head_seed=downstream_head_seed,
    ).to(activation_images.device)
    functional.eval()
    with torch.inference_mode():
        functional(activation_images)
    expected_layers = set(provider.layouts)
    if set(provider.weights_by_layer) != expected_layers:
        missing = sorted(expected_layers - set(provider.weights_by_layer))
        raise RuntimeError(f"sequential materialization missed layers: {missing}")
    model = ResNet18Slim(
        channels_in=model_config.channels_in,
        o_dim=model_config.num_classes,
        nlin=model_config.activation,
        dropout=model_config.dropout,
        init_type=None,
        width_mult=model_config.width_mult,
    ).to(activation_images.device)
    state = model.state_dict()
    for layer_key, weight in provider.weights_by_layer.items():
        state[f"{layer_key}.weight"] = weight.to(device=state[f"{layer_key}.weight"].device, dtype=state[f"{layer_key}.weight"].dtype)
    model.load_state_dict(state, strict=True)
    return model


class FlowEvaluationCandidateFactory:
    """Adapter from common ``Candidate`` payloads to trainable materialized models."""

    def __init__(
        self,
        *,
        loaded_flow: LoadedSealedFlow,
        decoder: TileDecoder,
        layouts: Sequence[LayerTileLayout],
        architecture_features: torch.Tensor,
        dataset_embedding: torch.Tensor,
        model_config: FunctionalResNetConfig,
        activation_images: torch.Tensor,
        activation_context_provenance: Mapping[str, object],
        tile_mask: torch.Tensor | None = None,
    ) -> None:
        if loaded_flow.seal.get("codec") != "ours":
            raise ValueError("tile candidate factory requires the sealed ours substrate")
        if loaded_flow.seal.get("activation_conditioning") is not True:
            raise ValueError("primary ours flow must be sealed with activation_conditioning=true")
        assert_decoder_matches_loaded_flow(loaded_flow, decoder)
        validate_context_provenance(activation_context_provenance, activation_images)
        self.loaded_flow = loaded_flow
        self.decoder = decoder
        self.layouts = tuple(layouts)
        self.architecture_features = architecture_features
        self.dataset_embedding = dataset_embedding
        self.model_config = model_config
        self.activation_images = activation_images
        self.activation_context_provenance = dict(activation_context_provenance)
        self.tile_mask = tile_mask

    def __call__(self, candidate: object) -> ResNet18Slim:
        payload = getattr(candidate, "payload", None)
        if not isinstance(payload, Mapping):
            raise TypeError("flow Candidate.payload must be a mapping")
        device = next(self.loaded_flow.model.parameters()).device
        anchors = payload.get("anchors")
        if anchors is not None:
            anchors = anchors.to(device)
        seed = int(payload.get("seed", 0))
        if payload.get("activation_conditioning", True) is not True:
            raise ValueError("candidate payload cannot disable sealed primary activation conditioning")
        if isinstance(self.decoder, nn.Module):
            self.decoder.to(device).eval().requires_grad_(False)
        requested_solver = payload.get("solver", self.loaded_flow.seal["solver"])
        requested_steps = int(payload.get("nfe_steps", self.loaded_flow.seal["nfe_steps"]))
        if requested_solver != self.loaded_flow.seal["solver"] or requested_steps != int(self.loaded_flow.seal["nfe_steps"]):
            raise ValueError("candidate solver/NFE must equal the sealed EMA validation selection")
        sampling = FlowSamplingConfig(
            path_kind=str(self.loaded_flow.seal["path_kind"]),
            solver=str(requested_solver),
            nfe_steps=requested_steps,
            seed=seed,
            activation_conditioning=True,
        )
        provider = FlowDecodedConvProvider(
            flow=self.loaded_flow.model,
            decoder=self.decoder,
            layouts=self.layouts,
            architecture_features=self.architecture_features.to(device),
            dataset_embedding=self.dataset_embedding.to(device),
            normalizer=self.loaded_flow.normalizer,
            config=sampling,
            anchors=anchors,
            tile_mask=None if self.tile_mask is None else self.tile_mask.to(device),
        )
        return materialize_standard_resnet(
            provider=provider,
            model_config=self.model_config,
            activation_images=self.activation_images.to(device),
            downstream_head_seed=seed,
        )


class WeightCLIPMultiWindowFlowEvaluationCandidateFactory:
    """Our multi-window extension over official tokens; never a released baseline."""

    def __init__(
        self,
        *,
        loaded_flow: LoadedSealedFlow,
        decoder: nn.Module,
        dataset_embedding: torch.Tensor,
        architecture_features: torch.Tensor,
        num_classes: int,
        width_mult: float = 0.5,
        paired_head_seed: int = 0,
        bn_calibration_batches: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if loaded_flow.seal.get("codec") != "weightclip":
            raise ValueError("WeightCLIP multi-window factory requires a sealed weightclip substrate")
        assert_decoder_matches_loaded_flow(loaded_flow, decoder)
        self.loaded_flow = loaded_flow
        self.decoder = decoder
        self.dataset_embedding = dataset_embedding
        self.architecture_features = architecture_features
        self.num_classes = int(num_classes)
        self.width_mult = float(width_mult)
        self.paired_head_seed = int(paired_head_seed)
        self.bn_calibration_batches = None if bn_calibration_batches is None else list(bn_calibration_batches)

    def __call__(self, candidate: object) -> ResNet18Slim:
        from big_vae.weightclip_benchmark.weightclip_full import materialize_weightclip_controlled_model

        payload = getattr(candidate, "payload", None)
        if not isinstance(payload, Mapping):
            raise TypeError("WeightCLIP flow Candidate.payload must be a mapping")
        device = next(self.loaded_flow.model.parameters()).device
        anchor = payload.get("anchors")
        template_code = payload.get("template_code")
        path_kind = str(self.loaded_flow.seal["path_kind"])
        if path_kind == "gaussian" and anchor is not None:
            raise ValueError("anchor-free WeightCLIP flow forbids oracle anchors")
        if path_kind == "paired_anchor" and anchor is None:
            raise ValueError("paired WeightCLIP flow requires exact clean per-window anchors")
        if path_kind == "paired_anchor" and template_code is not None:
            raise ValueError("paired WeightCLIP flow must project to z_enc, not a default template")
        if path_kind == "gaussian":
            if not torch.is_tensor(template_code):
                raise ValueError("anchor-free WeightCLIP flow requires architecture-default template_code")
            expected_template_sha = payload.get("template_code_sha256")
            if not isinstance(expected_template_sha, str) or context_tensor_sha256(template_code) != expected_template_sha:
                raise ValueError("anchor-free WeightCLIP template code is not hash-bound")
        requested_solver = payload.get("solver", self.loaded_flow.seal["solver"])
        requested_steps = int(payload.get("nfe_steps", self.loaded_flow.seal["nfe_steps"]))
        if requested_solver != self.loaded_flow.seal["solver"] or requested_steps != int(self.loaded_flow.seal["nfe_steps"]):
            raise ValueError("candidate solver/NFE must equal the sealed EMA validation selection")
        self.decoder.to(device).eval().requires_grad_(False)
        if anchor is not None:
            raw_anchor = anchor.to(device)
            base = self.loaded_flow.normalizer.normalize(raw_anchor)
        else:
            raw_anchor = None
            generator = torch.Generator(device=device).manual_seed(int(payload.get("seed", 0)))
            latent_dim = int(self.loaded_flow.model.config.latent_dim)
            base = torch.randn(
                (self.decoder.window_count, self.decoder.window_size, latent_dim),
                device=device,
                generator=generator,
            )
        expected = (self.decoder.window_count, self.decoder.window_size)
        if base.ndim != 3 or tuple(base.shape[:2]) != expected:
            raise ValueError(f"WeightCLIP flow code grouping {tuple(base.shape)} does not match official windows {expected}")
        dataset = self.dataset_embedding.to(device).reshape(1, -1).expand(self.decoder.window_count, -1)
        architecture = self.architecture_features.to(device)
        if tuple(architecture.shape[:2]) != (self.decoder.window_count, self.decoder.window_size):
            raise ValueError("WeightCLIP flow requires shared architecture features for every official window token")

        def velocity(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            value = self.loaded_flow.model(
                z,
                t,
                dataset_embedding=dataset,
                architecture_features=architecture,
                token_mask=self.decoder.body_token_mask.to(device),
            )
            value = value * self.decoder.body_token_mask.to(device).unsqueeze(-1).to(value.dtype)
            return value

        integration = integrate_flow(
            velocity,
            base,
            method=str(self.loaded_flow.seal["solver"]),
            steps=int(self.loaded_flow.seal["nfe_steps"]),
        )
        code = self.loaded_flow.normalizer.denormalize(integration.endpoint)
        if path_kind == "paired_anchor":
            assert raw_anchor is not None
            code = project_weightclip_paired_endpoint(code, raw_anchor, self.decoder.body_token_mask)
        else:
            assert torch.is_tensor(template_code)
            code = project_weightclip_controlled_endpoint(
                code,
                template_code.to(device),
                self.decoder.body_token_mask,
                reference_kind="architecture-default template",
            )
        return materialize_weightclip_controlled_model(
            self.decoder,
            code,
            num_classes=self.num_classes,
            width_mult=self.width_mult,
            paired_head_seed=self.paired_head_seed,
            bn_calibration_batches=self.bn_calibration_batches,
        )


def project_weightclip_paired_endpoint(
    endpoint: torch.Tensor,
    anchor: torch.Tensor,
    body_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Project paired WC transport so nonbody rows remain bitwise the clean anchor."""

    return project_weightclip_controlled_endpoint(
        endpoint, anchor, body_token_mask, reference_kind="clean paired anchor"
    )


def project_weightclip_controlled_endpoint(
    endpoint: torch.Tensor,
    reference: torch.Tensor,
    body_token_mask: torch.Tensor,
    *,
    reference_kind: str,
) -> torch.Tensor:
    """Keep generated body rows and restore every discarded row bitwise."""

    if endpoint.shape != reference.shape or tuple(body_token_mask.shape) != tuple(endpoint.shape[:2]):
        raise ValueError(f"WeightCLIP endpoint/{reference_kind}/body mask do not align")
    body = body_token_mask.to(device=endpoint.device, dtype=torch.bool).unsqueeze(-1)
    reference = reference.to(device=endpoint.device, dtype=endpoint.dtype)
    projected = torch.where(body, endpoint, reference)
    if not torch.equal(projected[~body_token_mask], reference[~body_token_mask]):
        raise RuntimeError(f"WeightCLIP projection failed to preserve nonbody {reference_kind} rows bitwise")
    return projected
