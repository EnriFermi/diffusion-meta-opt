"""Real, fail-closed Stage-G model and dataset factories."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import hashlib
import json
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from big_vae.flow_matching.sample import (
    FlowEvaluationCandidateFactory,
    WeightCLIPMultiWindowFlowEvaluationCandidateFactory,
    load_sealed_flow,
)
from big_vae.flow_matching.dataset import canonical_codec_fingerprint
from big_vae.flow_matching.features import retokenize_semantic_group_features
from big_vae.weightclip_benchmark.contract import DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.decoder_adapters import OfficialWeightCLIPMultiWindowDecoderAdapter
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    load_checkpoint_state,
    load_ood_conditioning_bundle,
    load_dataset_pt,
    load_sealed_ours_decoder,
    load_task_fit_bundle,
)
from big_vae.weightclip_benchmark.resnet18slim import ResNet18Slim
from big_vae.weightclip_benchmark.resnet_functional import FunctionalResNet18Slim, FunctionalResNetConfig
from big_vae.weightclip_benchmark.task_latent_fit import LatentDecodedConvProvider, LayerTileLayout
from big_vae.weightclip_benchmark.weightclip_full import materialize_weightclip_controlled_model, materialize_weightclip_native_model
from big_vae.weightclip_benchmark.official_bridge import OfficialWeightCLIPBridge
from big_vae.weightclip_benchmark.paired_sampling import ResettableEpochSampler


_OURS_DECODER_CACHE: dict[tuple[str, str], Any] = {}
_FLOW_CACHE: dict[tuple[str, str, str, str, str, str], Any] = {}
_OFFICIAL_BASE_CACHE: dict[tuple[str, str, str, str], Any] = {}


def _ref(value: Any, label: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a path/SHA/bytes artifact reference")
    reference = ArtifactRef.from_mapping(value)
    reference.verify(label)
    return reference


def _dataset_ref(dataset: str, config: Mapping[str, Any]) -> ArtifactRef:
    registry = config.get("datasets")
    if not isinstance(registry, Mapping) or dataset not in registry:
        raise ValueError(f"dataset {dataset!r} is absent from the sealed evaluation dataset registry")
    row = registry[dataset]
    reference = _ref(row.get("dataset_pt") if isinstance(row, Mapping) else None, f"dataset {dataset}")
    return reference


def build_loaders(*, dataset: str, config: Mapping[str, Any]) -> dict[str, DataLoader[Any]]:
    """Build only from a hash-bound official dataset.pt; never download/fallback."""

    reference = _dataset_ref(dataset, config)
    payload = load_dataset_pt(reference)
    evaluation = dict(config.get("evaluation", {}))
    batch_size = int(evaluation.get("batch_size", 128))
    workers = int(evaluation.get("num_workers", 0))
    seed = int(evaluation.get("loader_seed", 0))

    def loader(split: str, *, shuffle: bool) -> DataLoader[Any]:
        item = payload[split]
        tensor_dataset = TensorDataset(item.data, item.targets)
        sampler = ResettableEpochSampler(len(tensor_dataset), seed) if shuffle else None
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            pin_memory=bool(evaluation.get("pin_memory", True)),
            persistent_workers=workers > 0,
        )

    return {
        "train": loader("trainset", shuffle=True),
        "bn": loader("trainset", shuffle=False),
        "selection": loader("valset", shuffle=False),
        "test": loader("testset", shuffle=False),
    }


def _model_config(state: Mapping[str, torch.Tensor]) -> FunctionalResNetConfig:
    return FunctionalResNetConfig(
        width_mult=float(state["conv1.weight"].shape[0]) / 64.0,
        channels_in=int(state["conv1.weight"].shape[1]),
        num_classes=int(state["fc.weight"].shape[0]),
        dropout=DEFAULT_CONTRACT.population.dropout,
        head_policy="default_random_frozen",
        bn_policy="default_frozen",
    )


def _candidate_dataset_embedding(bundle: Mapping[str, Any], payload: Mapping[str, Any]) -> torch.Tensor:
    index = int(payload.get("prompt_candidate_index", 0))
    candidates = bundle.get("dataset_embedding_candidates")
    provenance = bundle.get("dataset_prompt_provenance", {})
    prompt_sets = provenance.get("candidate_image_indices")
    if candidates is None and bundle.get("dataset_embedding_bank") is not None:
        candidates = bundle["dataset_embedding_bank"]
        provenance = bundle.get("dataset_embedding_bank_provenance", {})
        prompt_sets = provenance.get("candidate_indices")
    if candidates is None:
        if index != 0:
            raise ValueError("bundle has no candidate embeddings but candidate requested a nonzero prompt index")
        return bundle["dataset_embedding"]
    if not torch.is_tensor(candidates) or not isinstance(prompt_sets, list) or index >= len(candidates):
        raise ValueError("conditioning bundle prompt candidate geometry is invalid")
    indices = prompt_sets[index]
    digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    if payload.get("prompt_indices") != indices or payload.get("prompt_set_sha256") != digest:
        raise ValueError("candidate prompt subset is not bound to the conditioning bundle")
    return candidates[index]


def _load_code(
    reference: ArtifactRef,
    *,
    expected_dataset: str,
    expected_codec: str,
    expected_mode: str,
) -> tuple[torch.Tensor, str]:
    payload = torch.load(reference.verify("latent code"), map_location="cpu", weights_only=False)
    code = payload.get("z_task", payload.get("code", payload)) if isinstance(payload, Mapping) else payload
    if not isinstance(payload, Mapping):
        raise ValueError("production latent code must include identity/provenance, not a bare tensor")
    observed_dataset = payload.get("dataset_id", payload.get("identity", {}).get("dataset_id"))
    observed_codec = payload.get("codec", payload.get("provenance", {}).get("codec"))
    if str(observed_dataset) != str(expected_dataset) or str(observed_codec) != str(expected_codec):
        raise ValueError(
            f"latent code identity mismatch: dataset/codec={observed_dataset}/{observed_codec}, "
            f"expected={expected_dataset}/{expected_codec}"
        )
    producer = payload.get("producer", {})
    if (
        producer.get("execution_status") != "complete"
        or producer.get("substrate_label") != "WeightCLIP codec + common-zoo full-window prior"
        or producer.get("mode") != expected_mode
        or producer.get("released_mapper_or_bank") is not False
    ):
        raise ValueError("latent code was not emitted by the declared executed common-zoo full-window producer")
    if not torch.is_tensor(code) or not torch.isfinite(code).all():
        raise ValueError("latent code artifact contains no finite code tensor")
    code_cpu = code.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(code_cpu.dtype).encode())
    digest.update(str(tuple(code_cpu.shape)).encode())
    digest.update(code_cpu.numpy().tobytes())
    if producer.get("code_tensor_sha256") != digest.hexdigest():
        raise ValueError("latent code tensor hash disagrees with producer ledger")
    fingerprint = payload.get("codec_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("latent code artifact lacks codec_fingerprint")
    return code.detach().float(), fingerprint


def _official_decoder(payload: Mapping[str, Any], bundle: Mapping[str, Any], config: Mapping[str, Any], device: str) -> Any:
    official = dict(config.get("official_weightclip", {}))
    checkpoint = _ref(payload.get("official_checkpoint"), "official WeightCLIP checkpoint")
    dataset_encoder = _ref(payload.get("official_dataset_encoder"), "official WeightCLIP dataset encoder")
    required_sha = str(official.get("checkpoint_sha256", ""))
    encoder_sha = str(official.get("dataset_encoder_sha256", ""))
    if checkpoint.sha256 != required_sha or dataset_encoder.sha256 != encoder_sha:
        raise ValueError("candidate official artifacts disagree with the pinned evaluation config")
    repo = Path(str(official.get("repo_path", ""))).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"pinned official WeightCLIP checkout is absent: {repo}")
    reference_key = "template_checkpoint" if bundle["bundle_kind"] == "ood_conditioning" else "source_checkpoint"
    reference = ArtifactRef.from_mapping(bundle[reference_key])
    source = load_checkpoint_state(reference)
    bridge = OfficialWeightCLIPBridge(
        str(repo),
        str(official.get("cache_dir", checkpoint.verify("official WeightCLIP checkpoint").parents[3])),
        device=device,
    )
    cache_key = (checkpoint.sha256, dataset_encoder.sha256, str(repo), device)
    loaded = _OFFICIAL_BASE_CACHE.get(cache_key)
    if loaded is None:
        loaded = bridge.load_codec_tokenizer_only(
            reference_state=source,
            reference_state_sha256=reference.sha256,
            checkpoint_path=checkpoint.path,
            dataset_encoder_path=dataset_encoder.path,
        )
        _OFFICIAL_BASE_CACHE[cache_key] = loaded
    elif loaded.provenance.get("reference_state_sha256") != reference.sha256:
        loaded = bridge.retarget_tokenizer(
            loaded,
            reference_state=source,
            reference_state_sha256=reference.sha256,
        )
    return OfficialWeightCLIPMultiWindowDecoderAdapter(
        loaded.weight_model,
        loaded.tokenizer,
        anchor_state=source,
        window_size=512,
        provenance={"codec": "weightclip", "checkpoint_sha256": checkpoint.sha256, "official": loaded.provenance},
    )


def _ours_decoder(payload: Mapping[str, Any], device: str) -> Any:
    seal = _ref(payload.get("codec_seal"), "ours codec seal")
    key = (seal.sha256, device)
    if key not in _OURS_DECODER_CACHE:
        _OURS_DECODER_CACHE[key] = load_sealed_ours_decoder(seal, device=device)
    return _OURS_DECODER_CACHE[key]


def _sealed_flow(payload: Mapping[str, Any], *, codec: str, device: str) -> Any:
    seal = _ref(payload.get("flow_seal"), "flow seal")
    checkpoint = _ref(payload.get("flow_checkpoint"), "flow checkpoint")
    normalizer = _ref(payload.get("normalizer"), "flow normalizer")
    path_kind = str(payload["path_kind"])
    key = (seal.sha256, checkpoint.sha256, normalizer.sha256, codec, path_kind, device)
    if key not in _FLOW_CACHE:
        _FLOW_CACHE[key] = load_sealed_flow(
            seal_path=seal.path,
            checkpoint_path=checkpoint.path,
            normalizer_path=normalizer.path,
            expected_codec=codec,
            expected_path_kind=path_kind,
            device=device,
        )
    return _FLOW_CACHE[key]


def _materialize_ours_reconstruction(payload: Mapping[str, Any], bundle: Mapping[str, Any], device: str) -> nn.Module:
    decoder = _ours_decoder(payload, device)
    codes = bundle["z_enc"].to(device)
    layouts = [LayerTileLayout(**item) for item in bundle["layouts"]]
    provider = LatentDecodedConvProvider(
        nn.Parameter(codes, requires_grad=False),
        decoder,
        layouts,
        bundle["architecture_features"].to(device),
        None if bundle.get("tile_mask") is None else bundle["tile_mask"].to(device),
        activation_conditioning=True,
    )
    source = load_checkpoint_state(ArtifactRef.from_mapping(bundle["source_checkpoint"]))
    model_config = _model_config(source)
    functional = FunctionalResNet18Slim(model_config, provider, source_state=None, random_head_seed=int(payload["seed"])).to(device)
    provider.begin_materialization()
    functional.eval()
    with torch.inference_mode():
        functional(bundle["context_images"].to(device))
    provider.finish_materialization()
    model = ResNet18Slim(
        channels_in=model_config.channels_in,
        o_dim=model_config.num_classes,
        width_mult=model_config.width_mult,
        dropout=model_config.dropout,
        init_type=None,
    ).to(device)
    state = model.state_dict()
    for key, value in provider.cached_weights.items():
        state[f"{key}.weight"] = value.to(state[f"{key}.weight"])
    model.load_state_dict(state, strict=True)
    return model


def build_model(*, payload: Mapping[str, Any], config: Mapping[str, Any]) -> nn.Module:
    """Materialize one candidate with all artifact and conditioning gates active."""

    if not isinstance(payload, Mapping):
        raise TypeError("candidate payload must be a mapping")
    kind = str(payload.get("method_kind", ""))
    device = str(config.get("runtime", {}).get("device", "cuda"))
    seed = int(payload.get("seed", -1))
    if seed < 0:
        raise ValueError("candidate payload requires a non-negative paired seed")
    dropout = float(config.get("evaluation", {}).get("model_dropout", -1.0))
    if dropout != DEFAULT_CONTRACT.population.dropout:
        raise ValueError(f"evaluation dropout must remain {DEFAULT_CONTRACT.population.dropout}, got {dropout}")
    if kind == "scratch":
        classes = int(payload["num_classes"])
        return ResNet18Slim(o_dim=classes, width_mult=float(payload.get("width_mult", 0.5)), dropout=dropout, init_type="kaiming_uniform")
    if kind in {"anchor", "anchor_untouched"}:
        state = load_checkpoint_state(_ref(payload.get("checkpoint"), "anchor checkpoint"))
        model = ResNet18Slim(o_dim=int(state["fc.weight"].shape[0]), width_mult=float(state["conv1.weight"].shape[0]) / 64, dropout=dropout, init_type=None)
        model.load_state_dict(state, strict=True)
        return model
    if kind not in {
        "ours_flow", "ours_flow_oracle_anchor", "ours_reconstruction", "weightclip_flow", "weightclip_flow_oracle_anchor", "weightclip_controlled",
        "weightclip_commonzoo_fullwindow_ridge", "weightclip_commonzoo_fullwindow_memory", "weightclip_commonzoo_fullwindow_nearest_code", "weightclip_commonzoo_fullwindow_ridge_native_oracle",
        "weightclip_commonzoo_fullwindow_memory_native_oracle", "weightclip_commonzoo_fullwindow_nearest_code_native_oracle",
    }:
        raise ValueError(f"unsupported method_kind {kind!r}")
    path_kind = str(payload.get("path_kind", ""))
    if kind in {"ours_flow", "weightclip_flow", "ours_flow_oracle_anchor", "weightclip_flow_oracle_anchor"} and path_kind == "gaussian":
        if "task_bundle" in payload or "anchors" in payload:
            raise ValueError("anchor-free flow forbids task-fit bundles and target anchors")
        bundle_ref = _ref(payload.get("conditioning_bundle"), "OOD conditioning bundle")
        bundle = load_ood_conditioning_bundle(bundle_ref.path)
    elif kind in {"ours_flow", "weightclip_flow", "ours_flow_oracle_anchor", "weightclip_flow_oracle_anchor"} and path_kind == "paired_anchor":
        if "conditioning_bundle" in payload:
            raise ValueError("paired/oracle flow requires a target task-fit bundle, not OOD-only conditioning")
        bundle_ref = _ref(payload.get("task_bundle"), "task-fit bundle")
        bundle = load_task_fit_bundle(bundle_ref.path)
    elif kind == "ours_reconstruction":
        bundle_ref = _ref(payload.get("task_bundle"), "task-fit bundle")
        bundle = load_task_fit_bundle(bundle_ref.path, expected_kind="ours_task_fit")
    else:
        if "task_bundle" in payload:
            raise ValueError("controlled OOD WeightCLIP modes forbid target task-fit bundles")
        bundle_ref = _ref(payload.get("conditioning_bundle"), "OOD conditioning bundle")
        bundle = load_ood_conditioning_bundle(bundle_ref.path)
    if str(bundle["identity"]["dataset_id"]) != str(payload.get("dataset")):
        raise ValueError("candidate dataset disagrees with task-fit bundle identity")
    state_ref_key = "template_checkpoint" if bundle["bundle_kind"] == "ood_conditioning" else "source_checkpoint"
    source = load_checkpoint_state(ArtifactRef.from_mapping(bundle[state_ref_key]))
    if kind == "ours_reconstruction":
        if bundle["bundle_kind"] != "ours_task_fit":
            raise ValueError("ours reconstruction requires an ours task-fit bundle")
        return _materialize_ours_reconstruction(payload, bundle, device)
    if kind in {"ours_flow", "ours_flow_oracle_anchor"}:
        if bundle["bundle_kind"] not in {"ours_task_fit", "ood_conditioning"}:
            raise ValueError("ours flow received the wrong codec/bundle kind")
        if payload.get("activation_conditioning") is not True:
            raise ValueError("ours flow requires activation conditioning")
        codec = _ours_decoder(payload, device)
        loaded = _sealed_flow(payload, codec="ours", device=device)
        ours_features = bundle.get("architecture_features", bundle.get("ours_architecture_features"))
        if bundle["bundle_kind"] == "ood_conditioning":
            ours_features = retokenize_semantic_group_features(
                ours_features,
                int(codec.model.cfg.big_vae.num_latents),
            )
        factory = FlowEvaluationCandidateFactory(
            loaded_flow=loaded,
            decoder=codec,
            layouts=[LayerTileLayout(**item) for item in bundle.get("layouts", bundle.get("ours_layouts"))],
            architecture_features=ours_features,
            dataset_embedding=_candidate_dataset_embedding(bundle, payload),
            model_config=_model_config(source),
            activation_images=bundle["context_images"],
            activation_context_provenance=bundle["context_provenance"],
            tile_mask=(
                bundle.get("tile_mask")
                if bundle["bundle_kind"] != "ood_conditioning"
                else torch.ones(
                    (len(bundle["ours_architecture_features"]), int(codec.model.cfg.big_vae.num_latents)), dtype=torch.bool
                )
            ),
        )
        call_payload = dict(payload)
        if path_kind == "paired_anchor":
            call_payload["anchors"] = bundle["z_enc"]
        elif "anchors" in call_payload:
            raise ValueError("anchor-free ours flow cannot carry anchors")
        return factory(SimpleNamespace(payload=call_payload))
    if bundle["bundle_kind"] not in {"weightclip_task_fit", "ood_conditioning"}:
        raise ValueError("WeightCLIP flow/mode received the wrong codec/bundle kind")
    decoder = _official_decoder(payload, bundle, config, device)
    wc_features = bundle.get("architecture_features", bundle.get("weightclip_architecture_features"))
    if not torch.is_tensor(wc_features):
        raise ValueError("conditioning bundle was not sealed with official WeightCLIP window geometry")
    if kind in {"weightclip_flow", "weightclip_flow_oracle_anchor"}:
        loaded = _sealed_flow(payload, codec="weightclip", device=device)
        factory = WeightCLIPMultiWindowFlowEvaluationCandidateFactory(
            loaded_flow=loaded,
            decoder=decoder,
            dataset_embedding=_candidate_dataset_embedding(bundle, payload),
            architecture_features=wc_features,
            num_classes=int(source["fc.weight"].shape[0]),
            width_mult=float(source["conv1.weight"].shape[0]) / 64,
            paired_head_seed=seed,
        )
        call_payload = dict(payload)
        if path_kind == "paired_anchor":
            call_payload["anchors"] = bundle["z_enc"]
        else:
            if "anchors" in call_payload:
                raise ValueError("anchor-free WeightCLIP flow cannot carry anchors")
            template_code = bundle.get("weightclip_template_z_enc")
            template_mask = bundle.get("weightclip_template_token_mask")
            if not torch.is_tensor(template_code) or not torch.is_tensor(template_mask):
                raise ValueError("anchor-free WeightCLIP flow requires an architecture-default template code")
            decoder_mask = decoder.window_token_mask.detach().cpu().bool()
            if not torch.equal(template_mask.detach().cpu().bool(), decoder_mask):
                raise ValueError("WeightCLIP conditioning template token mask disagrees with runtime decoder")
            call_payload["template_code"] = template_code
            call_payload["template_code_sha256"] = bundle["provenance"]["weightclip_template_code_sha256"]
        return factory(SimpleNamespace(payload=call_payload))
    code, code_fingerprint = _load_code(
        _ref(payload.get("code"), f"{kind} code"),
        expected_dataset=str(payload["dataset"]),
        expected_codec="weightclip",
        expected_mode={
            "weightclip_commonzoo_fullwindow_ridge": "ridge",
            "weightclip_commonzoo_fullwindow_memory": "memory",
            "weightclip_commonzoo_fullwindow_nearest_code": "nearest_code",
            "weightclip_commonzoo_fullwindow_ridge_native_oracle": "ridge_native_oracle",
            "weightclip_commonzoo_fullwindow_memory_native_oracle": "memory_native_oracle",
            "weightclip_commonzoo_fullwindow_nearest_code_native_oracle": "nearest_code_native_oracle",
        }[kind],
    )
    code_payload = torch.load(_ref(payload.get("code"), f"{kind} code").path, map_location="cpu", weights_only=False)
    code_producer = code_payload.get("producer", {})
    if (
        int(code_producer.get("prompt_candidate_index", -1)) != int(payload.get("prompt_candidate_index", 0))
        or code_producer.get("prompt_set_sha256") != payload.get("prompt_set_sha256")
    ):
        raise ValueError("WeightCLIP code prompt subset disagrees with the paired candidate payload")
    actual_fingerprint = canonical_codec_fingerprint("weightclip", decoder.provenance)
    if code_fingerprint != actual_fingerprint:
        raise ValueError("WeightCLIP latent code was produced by a different decoder substrate")
    if kind in {
        "weightclip_commonzoo_fullwindow_ridge_native_oracle",
        "weightclip_commonzoo_fullwindow_memory_native_oracle",
        "weightclip_commonzoo_fullwindow_nearest_code_native_oracle",
    }:
        return materialize_weightclip_native_model(decoder, code, num_classes=int(source["fc.weight"].shape[0]), width_mult=float(source["conv1.weight"].shape[0]) / 64, dropout=dropout)
    return materialize_weightclip_controlled_model(
        decoder,
        code,
        num_classes=int(source["fc.weight"].shape[0]),
        width_mult=float(source["conv1.weight"].shape[0]) / 64,
        dropout=dropout,
        paired_head_seed=seed,
        bn_calibration_batches=None,
    )
