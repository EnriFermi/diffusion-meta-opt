#!/usr/bin/env python3
"""Run frozen E4 held-out-lineage solver/NFE and decoded-health selection."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from big_vae.flow_matching.dataset import GroupedLatentRecord, LatentNormalizer, validate_record_codec_fingerprints
from big_vae.flow_matching.features import layer_role
from big_vae.flow_matching.model import ConditionalVelocityTransformer, FlowModelConfig
from big_vae.flow_matching.sample import project_weightclip_controlled_endpoint
from big_vae.weightclip_benchmark.latent_bundles import (
    ArtifactRef,
    assert_decoder_provenance_matches_bundle,
    load_checkpoint_state,
    load_sealed_ours_decoder,
    load_task_fit_bundle,
)
from big_vae.weightclip_benchmark.decoder_adapters import OfficialWeightCLIPMultiWindowDecoderAdapter
from big_vae.weightclip_benchmark.official_bridge import OfficialWeightCLIPBridge
from big_vae.weightclip_benchmark.resnet_functional import FunctionalResNet18Slim, FunctionalResNetConfig
from big_vae.weightclip_benchmark.task_latent_fit import (
    LatentDecodedConvProvider,
    LayerTileLayout,
    context_tensor_sha256,
)


SOLVER_NFE_GRID = (("euler", 4), ("euler", 8), ("euler", 16), ("euler", 32), ("heun", 4), ("heun", 8), ("heun", 16), ("heun", 32))
DECODE_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable E4 report conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _trajectory(
    model: torch.nn.Module,
    base: torch.Tensor,
    *,
    solver: str,
    steps: int,
    dataset_embedding: torch.Tensor,
    architecture_features: torch.Tensor,
    token_mask: torch.Tensor,
    paired: bool,
    controlled_body_only: bool,
    nonbody_template: torch.Tensor | None = None,
) -> dict[float, torch.Tensor]:
    if steps % 4:
        raise ValueError("E4 NFE step count must make quarter-time decode points exact")
    z = base.clone()
    body = architecture_features[..., 1].bool().unsqueeze(-1)
    if nonbody_template is not None:
        if nonbody_template.shape != z.shape:
            raise ValueError("E4 nonbody template/code geometry mismatch")
        z = torch.where(body, z, nonbody_template)
    outputs = {0.0: z.clone()}
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full((len(z),), index * dt, device=z.device)

        def velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            result = model(
                value,
                time,
                dataset_embedding=dataset_embedding,
                architecture_features=architecture_features,
                token_mask=(body.squeeze(-1) if controlled_body_only else token_mask),
            )
            return result * body.to(result.dtype) if controlled_body_only else result

        first = velocity(z, t0)
        if solver == "euler":
            z = z + dt * first
        elif solver == "heun":
            predicted = z + dt * first
            t1 = torch.full((len(z),), (index + 1) * dt, device=z.device)
            z = z + 0.5 * dt * (first + velocity(predicted, t1))
        else:
            raise ValueError(f"unsupported E4 solver {solver!r}")
        if paired:
            z = torch.where(body, z, base)
        elif nonbody_template is not None:
            z = torch.where(body, z, nonbody_template)
        time = (index + 1) / steps
        if time in DECODE_TIMES:
            outputs[time] = z.clone()
    if set(outputs) != set(DECODE_TIMES):
        raise RuntimeError("E4 trajectory missed a frozen decode time")
    return outputs


def _accumulate_role_health(accumulator: dict[str, dict[str, float]], state: Mapping[str, torch.Tensor]) -> None:
    for key, value in state.items():
        role = layer_role(key)
        if role not in {"stem", "residual_conv1", "residual_conv2", "projection", "batchnorm"}:
            continue
        row = accumulator.setdefault(role, {"count": 0.0, "elements": 0.0, "sum_squares": 0.0, "finite": 0.0})
        tensor = value.detach().float()
        row["count"] += 1
        row["elements"] += tensor.numel()
        row["sum_squares"] += float(tensor.square().sum().item())
        row["finite"] += float(torch.isfinite(tensor).sum().item())


def _finalize_role_health(accumulator: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    return {
        role: {
            "count": int(row["count"]),
            "elements": int(row["elements"]),
            "weight_rms": (row["sum_squares"] / max(row["elements"], 1.0)) ** 0.5,
            "finite_fraction": row["finite"] / max(row["elements"], 1.0),
        }
        for role, row in accumulator.items()
    }


def _decode_ours(
    code: torch.Tensor,
    bundle: Mapping[str, Any],
    decoder: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    source = load_checkpoint_state(ArtifactRef.from_mapping(bundle["source_checkpoint"]))
    layouts = [LayerTileLayout(**item) for item in bundle["layouts"]]
    provider = LatentDecodedConvProvider(
        torch.nn.Parameter(code, requires_grad=False),
        decoder,
        layouts,
        bundle["architecture_features"].to(device),
        bundle["tile_mask"].to(device),
        activation_conditioning=True,
    )
    config = FunctionalResNetConfig(
        width_mult=float(source["conv1.weight"].shape[0]) / 64.0,
        channels_in=int(source["conv1.weight"].shape[1]),
        num_classes=int(source["fc.weight"].shape[0]),
        dropout=0.15,
        head_policy="original_frozen_source",
        bn_policy="original_frozen_source",
    )
    model = FunctionalResNet18Slim(
        config,
        provider,
        source_state={key: value.to(device) for key, value in source.items()},
        random_head_seed=0,
    ).to(device).eval()
    provider.begin_materialization()
    with torch.no_grad():
        model(bundle["context_images"].to(device))
    provider.finish_materialization()
    return {f"{key}.weight": value for key, value in provider.cached_weights.items()}


def _wc_decoders_once(
    raw: Mapping[str, Any], bundles: list[Mapping[str, Any]], device: str
) -> dict[str, OfficialWeightCLIPMultiWindowDecoderAdapter]:
    """Load the learned 9 GB codec once and retarget only its sparse tokenizer."""

    decoder_cfg = raw["decoder"]
    kwargs = dict(decoder_cfg.get("kwargs", {}))
    bridge = OfficialWeightCLIPBridge(
        str(kwargs["official_repo"]),
        str(kwargs["cache_dir"]),
        device=device,
    )
    loaded = None
    decoders: dict[str, OfficialWeightCLIPMultiWindowDecoderAdapter] = {}
    for bundle in bundles:
        source_ref = ArtifactRef.from_mapping(bundle["source_checkpoint"])
        source = load_checkpoint_state(source_ref)
        if loaded is None:
            loaded = bridge.load_codec_tokenizer_only(
                reference_state=source,
                reference_state_sha256=source_ref.sha256,
                checkpoint_path=decoder_cfg["checkpoint"],
                dataset_encoder_path=kwargs["dataset_encoder_path"],
                allow_download=False,
            )
        else:
            loaded = bridge.retarget_tokenizer(
                loaded,
                reference_state=source,
                reference_state_sha256=source_ref.sha256,
            )
        decoder = OfficialWeightCLIPMultiWindowDecoderAdapter(
            loaded.weight_model,
            loaded.tokenizer,
            anchor_state=source,
            window_size=int(kwargs.get("window_size", 512)),
            provenance={
                "codec": "weightclip",
                "checkpoint_sha256": loaded.provenance["checkpoint"]["sha256"],
                "official": loaded.provenance,
            },
        )
        assert_decoder_provenance_matches_bundle(
            decoder.provenance,
            bundle["provenance"]["decoder"],
            codec="weightclip",
        )
        group_id = str(bundle["identity"]["group_id"])
        if group_id in decoders:
            raise ValueError(f"duplicate E4 task bundle group_id {group_id!r}")
        decoders[group_id] = decoder
    return decoders


def _unique_expanded_paths(patterns: list[str]) -> list[Path]:
    return sorted({Path(path).resolve() for pattern in patterns for path in glob.glob(pattern)})


def _validate_expected_inventory(
    records: list[GroupedLatentRecord], bundles: list[Mapping[str, Any]], inventory_path: Path
) -> None:
    inventory = json.loads(inventory_path.read_text())
    expected = set(map(str, inventory.get("group_ids", [])))
    actual = {record.group_id for record in records}
    if len(actual) != len(records):
        raise ValueError("E4 validation records contain duplicate group IDs")
    bundle_ids = [str(bundle["identity"]["group_id"]) for bundle in bundles]
    if len(set(bundle_ids)) != len(bundle_ids):
        raise ValueError("E4 task bundles contain duplicate group IDs")
    if not expected or actual != expected or set(bundle_ids) != expected:
        raise ValueError("E4 records/bundles do not exactly match the sealed validation inventory")
    datasets: dict[str, set[str]] = {}
    for record in records:
        datasets.setdefault(record.dataset_id, set()).add(record.lineage_id)
    expected_dataset_count = int(inventory.get("expected_dataset_count", 10))
    expected_lineages = int(inventory.get("expected_lineages_per_dataset", 7))
    if len(datasets) != expected_dataset_count or any(len(rows) != expected_lineages for rows in datasets.values()):
        raise ValueError("E4 validation inventory is not the declared 10-dataset x 7-lineage balanced grid")


def write_e4_selection_report(
    *,
    output: Path,
    codec: str,
    codec_fingerprint: str,
    checkpoint_path: Path,
    normalizer_path: Path,
    sweep: list[dict[str, Any]],
    validation_record_artifacts: list[dict[str, Any]],
    task_bundle_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_grid = {(solver, steps) for solver, steps in SOLVER_NFE_GRID}
    observed_grid = {(str(row["solver"]), int(row["nfe_steps"])) for row in sweep}
    if observed_grid != expected_grid:
        raise ValueError("E4 report requires the complete frozen solver/NFE grid")
    if not any(row["decoded_finite_fraction"] == 1.0 for row in sweep):
        raise RuntimeError("no E4 solver/NFE candidate decoded finitely")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = {
        "schema_version": 1,
        "kind": "flow_decoded_e4_sweep",
        "codec": codec,
        "path_kind": checkpoint["train_config"]["path_kind"],
        "codec_fingerprint": codec_fingerprint,
        "checkpoint_sha256": _sha(checkpoint_path),
        "normalizer_sha256": _sha(normalizer_path),
        "validation_record_artifacts": validation_record_artifacts,
        "task_bundle_artifacts": task_bundle_artifacts,
        "selection_policy": "no_per_arm_selection; consumed by one common four-arm pre-OOD selector",
        "frozen_grid": {"solvers": ["euler", "heun"], "nfe_steps": [4, 8, 16, 32], "decode_times": list(DECODE_TIMES)},
        "sweep": sweep,
    }
    _immutable_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    codec = str(raw["codec"])
    device = torch.device(str(raw.get("device", "cuda")))
    checkpoint_path = Path(raw["checkpoint"]).resolve()
    normalizer_path = Path(raw["normalizer"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("schema_version", -1)) != 3:
        raise ValueError("E4 requires immutable-contract flow checkpoint v3")
    record_paths = _unique_expanded_paths(list(raw["validation_record_globs"]))
    records = [GroupedLatentRecord.load(path) for path in record_paths]
    records = [record for record in records if record.split == "validation"]
    fingerprint = validate_record_codec_fingerprints(records, expected_codec=codec)
    if checkpoint["provenance"]["codec_fingerprint"] != fingerprint:
        raise ValueError("E4 records and checkpoint codec fingerprints differ")
    normalizer = LatentNormalizer.from_state_dict(torch.load(normalizer_path, map_location="cpu", weights_only=False))
    if normalizer.codec_fingerprint != fingerprint:
        raise ValueError("E4 normalizer and records codec fingerprints differ")
    model = ConditionalVelocityTransformer(FlowModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["ema"]["shadow"], strict=True)
    model.to(device).eval().requires_grad_(False)
    bundle_paths = _unique_expanded_paths(list(raw["task_bundle_globs"]))
    bundles = [load_task_fit_bundle(path) for path in bundle_paths]
    inventory_path = Path(str(raw["expected_validation_inventory"])).resolve()
    _validate_expected_inventory(records, bundles, inventory_path)
    bundle_by_group = {bundle["identity"]["group_id"]: bundle for bundle in bundles}
    if set(record.group_id for record in records) - set(bundle_by_group):
        raise ValueError("E4 lacks task bundles for held-out validation records")
    ours_decoder = None
    if codec == "ours":
        ours_decoder = load_sealed_ours_decoder(ArtifactRef.create(raw["decoder"]["codec_seal"]), device=str(device))
    wc_decoders = {} if codec == "ours" else _wc_decoders_once(raw, bundles, str(device))
    sweep: list[dict[str, Any]] = []
    for solver, steps in SOLVER_NFE_GRID:
        endpoint_sq = 0.0
        endpoint_elements = 0
        decoded_by_time = {time: {} for time in DECODE_TIMES}
        decoded_models = 0
        decoded_finite = 0
        decoded_elements = 0
        for record in records:
            bundle = bundle_by_group[record.group_id]
            z_target = normalizer.normalize(record.z_task.to(device))
            if checkpoint["train_config"]["path_kind"] == "paired_anchor":
                if record.z_enc is None:
                    raise ValueError("paired E4 record lacks z_enc")
                base = normalizer.normalize(record.z_enc.to(device))
                paired = True
            else:
                seed = int.from_bytes(hashlib.sha256(record.group_id.encode()).digest()[:8], "little") % (2**63 - 1)
                generator = torch.Generator(device=device).manual_seed(seed)
                base = torch.randn(z_target.shape, generator=generator, device=device, dtype=z_target.dtype)
                paired = False
            embedding = record.dataset_embedding.to(device).unsqueeze(0).expand(len(z_target), -1)
            features = record.architecture_features.to(device)
            mask = record.tile_mask.to(device)
            wc_template = None
            if codec == "weightclip" and not paired:
                wc_template = bundle.get("weightclip_template_z_enc")
                expected_template_sha = bundle.get("provenance", {}).get("weightclip_template_code_sha256")
                if not torch.is_tensor(wc_template) or context_tensor_sha256(wc_template) != expected_template_sha:
                    raise ValueError(
                        "Gaussian WeightCLIP E4 requires a hash-bound architecture-default template code"
                    )
            trajectory = _trajectory(
                model,
                base,
                solver=solver,
                steps=steps,
                dataset_embedding=embedding,
                architecture_features=features,
                token_mask=mask,
                paired=paired,
                controlled_body_only=codec == "weightclip",
                nonbody_template=(
                    None
                    if wc_template is None
                    else normalizer.normalize(wc_template.to(device))
                ),
            )
            endpoint = trajectory[1.0]
            metric_mask = mask
            if codec == "weightclip":
                metric_mask = metric_mask & features[..., 1].to(dtype=torch.bool)
            expanded_mask = metric_mask.unsqueeze(-1)
            endpoint_sq += float(((endpoint - z_target).square() * expanded_mask).sum().item())
            endpoint_elements += int(expanded_mask.sum().item() * endpoint.shape[-1])
            wc_decoder = None if codec == "ours" else wc_decoders[record.group_id]
            for time, normalized_code in trajectory.items():
                code = normalizer.denormalize(normalized_code)
                if codec == "weightclip" and not paired:
                    assert torch.is_tensor(wc_template)
                    code = project_weightclip_controlled_endpoint(
                        code,
                        wc_template.to(device),
                        wc_decoder.body_token_mask,
                        reference_kind="architecture-default template",
                    )
                state = (
                    _decode_ours(code, bundle, ours_decoder, device)
                    if codec == "ours"
                    else wc_decoder.decode_state(code)
                )
                _accumulate_role_health(decoded_by_time[time], state)
                decoded_models += 1
                for value in state.values():
                    decoded_finite += int(torch.isfinite(value).sum().item())
                    decoded_elements += value.numel()
        sweep.append(
            {
                "solver": solver,
                "nfe_steps": steps,
                "actual_nfe": steps * (2 if solver == "heun" else 1),
                "endpoint_rmse": (endpoint_sq / max(endpoint_elements, 1)) ** 0.5,
                "decoded_models": decoded_models,
                "decoded_finite_fraction": decoded_finite / max(decoded_elements, 1),
                "decoded_health_by_time": {
                    str(time): _finalize_role_health(rows) for time, rows in decoded_by_time.items()
                },
            }
        )
    output = Path(raw["output_path"]).resolve()
    write_e4_selection_report(
        output=output,
        codec=codec,
        codec_fingerprint=fingerprint,
        checkpoint_path=checkpoint_path,
        normalizer_path=normalizer_path,
        sweep=sweep,
        validation_record_artifacts=[
            {"path": str(Path(path).resolve()), "sha256": _sha(Path(path)), "bytes": Path(path).stat().st_size}
            for path in record_paths
        ],
        task_bundle_artifacts=[
            {"path": str(Path(path).resolve()), "sha256": _sha(Path(path)), "bytes": Path(path).stat().st_size}
            for path in bundle_paths
        ],
    )
    print(json.dumps({"stage": "e4_sweep_complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
