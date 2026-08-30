#!/usr/bin/env python3
"""Execute common-zoo full-window WeightCLIP ridge/memory/nearest-code priors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef, load_ood_conditioning_bundle
from big_vae.weightclip_benchmark.official_bridge import OfficialWeightCLIPBridge
from big_vae.weightclip_benchmark.manifests import write_json_immutable


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_memory_fullwindow_batch(*, batch_size: int, sequence_length: int, source_count: int) -> dict[str, int]:
    """Hard-stop the official mapper's pre-subsample full-sequence OOM path."""

    if batch_size < 1 or batch_size > 4:
        raise ValueError(
            "full-window memory mapper batch_size must be 1..4 because official code allocates "
            "[batch,sequence,source_count] logits before token_subsample"
        )
    logits_bytes_fp32 = int(batch_size) * int(sequence_length) * int(source_count) * 4
    return {
        "microbatch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "source_count": int(source_count),
        "estimated_full_logits_bytes_fp32": logits_bytes_fp32,
    }


def produce_fullwindow_codes(
    *,
    mode: str,
    code_bank: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    entrypoint: Any,
    official_utils: Any,
    device: str,
    memory_kwargs: Mapping[str, Any] | None = None,
    producer_artifacts: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if mode not in {
        "ridge", "nearest_code", "memory", "ridge_native_oracle",
        "memory_native_oracle", "nearest_code_native_oracle",
    }:
        raise ValueError(f"unsupported full-window producer mode {mode!r}")
    if code_bank.get("kind") != "weightclip_commonzoo_fullwindow_code_bank":
        raise ValueError("producer requires the explicit common-zoo full-window code bank")
    fingerprint = str(code_bank["codec_fingerprint"])
    x_train = code_bank["X_dataset_embeddings"].float()
    y_train = code_bank["Y_fullwindow_codes"].float()
    if x_train.ndim != 2 or y_train.ndim != 3 or len(x_train) != len(y_train):
        raise ValueError("full-window code bank X/Y geometry is invalid")
    windows = int(code_bank["window_count"])
    window_size = int(code_bank["window_size"])
    latent_dim = int(code_bank["latent_dim"])
    if tuple(y_train.shape[1:]) != (windows * window_size, latent_dim):
        raise ValueError("full-window bank ledger disagrees with Y")
    algorithm = {
        "ridge_native_oracle": "ridge",
        "memory_native_oracle": "memory",
        "nearest_code_native_oracle": "nearest_code",
    }.get(mode, mode)
    ridge_state = None
    mapper = None
    producer_ledger: dict[str, Any] = {}
    if algorithm == "ridge":
        ridge_state = entrypoint.fit_direct_decoder(x_train, y_train, device=device)
        if producer_artifacts is not None:
            producer_artifacts["ridge_state"] = ridge_state
        producer_ledger = {
            "resolved_fit_kwargs": {},
            "fit_device": str(device),
            "fit_input_dtype": str(x_train.dtype),
            "fit_target_dtype": str(y_train.dtype),
        }
    elif algorithm == "memory":
        kwargs = dict(memory_kwargs or {})
        seed = int(kwargs.pop("seed", 0))
        temperature = float(kwargs.pop("temperature", 0.5))
        top_k = int(kwargs.pop("top_k", 32))
        memory_safety = validate_memory_fullwindow_batch(
            batch_size=int(kwargs.get("batch_size", 1)),
            sequence_length=int(y_train.shape[1]),
            source_count=int(len(y_train)),
        )
        python_state, numpy_state = random.getstate(), np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        cuda_devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(seed)
                mapper, best_loss, history = official_utils.train_memory_bank_translator(
                    x_train, y_train, device=device, **kwargs
                )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        mapper_state = {key: value.detach().cpu() for key, value in mapper.state_dict().items()}
        if producer_artifacts is not None:
            producer_artifacts["memory_mapper_state"] = mapper_state
        producer_ledger = {
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "resolved_training_kwargs": kwargs,
            "best_loss": float(best_loss),
            "history_length": len(history),
            "official_pre_subsample_allocation_gate": memory_safety,
        }
    outputs: list[dict[str, Any]] = []
    for target in targets:
        if target.get("weightclip_codec_fingerprint") != fingerprint:
            raise ValueError("target conditioning bundle and code bank use different WeightCLIP codecs")
        embeddings = target.get("dataset_embedding_candidates")
        if embeddings is None:
            embeddings = target["dataset_embedding"].reshape(1, -1)
        embeddings = embeddings.float()
        if embeddings.ndim != 2:
            raise ValueError("target prompt embeddings must be [candidates,embedding_dim]")
        if algorithm == "ridge":
            assert ridge_state is not None
            predictions = entrypoint.apply_direct_decoder(embeddings, ridge_state, device=device)
        elif algorithm == "nearest_code":
            predictions, stats = entrypoint.retrieve_neighbour_embeddings(
                embeddings, x_train, y_train, metric="cosine"
            )
            producer_ledger = {"metric": "cosine", **dict(stats)}
        else:
            assert mapper is not None
            predictions = entrypoint._sample_memory_bank_embeddings(
                mapper,
                embeddings,
                len(embeddings),
                device,
                producer_ledger["temperature"],
                producer_ledger["top_k"],
            )
        if tuple(predictions.shape[1:]) != (windows * window_size, latent_dim):
            raise ValueError(f"full-window prior changed native geometry: {tuple(predictions.shape)}")
        for prompt_index, prediction in enumerate(predictions):
            code = prediction.detach().cpu().reshape(windows, window_size, latent_dim)
            candidate_sets = target.get("dataset_prompt_provenance", {}).get("candidate_image_indices")
            prompt_indices = None if candidate_sets is None else candidate_sets[prompt_index]
            prompt_set_sha256 = (
                None
                if prompt_indices is None
                else hashlib.sha256(json.dumps(prompt_indices, separators=(",", ":")).encode()).hexdigest()
            )
            outputs.append(
                {
                    "schema_version": 1,
                    "dataset_id": str(target["identity"]["dataset_id"]),
                    "codec": "weightclip",
                    "codec_fingerprint": fingerprint,
                    "code": code,
                    "producer": {
                        "execution_status": "complete",
                        "substrate_label": "WeightCLIP codec + common-zoo full-window prior",
                        "mode": mode,
                        "algorithm": algorithm,
                        "prompt_candidate_index": prompt_index,
                        "prompt_indices": prompt_indices,
                        "prompt_set_sha256": prompt_set_sha256,
                        "dataset_embedding_sha256": _tensor_sha256(embeddings[prompt_index]),
                        "code_tensor_sha256": _tensor_sha256(code),
                        "window_count": windows,
                        "window_size": window_size,
                        "fit_once_on_concatenated_full_sequence": True,
                        "concat_then_single_detokenize": True,
                        "released_mapper_or_bank": False,
                        "exact_paper_mode_available": False,
                        **producer_ledger,
                    },
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bank_ref = ArtifactRef.create(raw["code_bank"])
    code_bank = torch.load(bank_ref.path, map_location="cpu", weights_only=False)
    target_refs = [ArtifactRef.create(path) for path in raw["conditioning_bundles"]]
    targets = [load_ood_conditioning_bundle(reference.path) for reference in target_refs]
    official = raw["official_weightclip"]
    bridge = OfficialWeightCLIPBridge(official["repo_path"], official["cache_dir"], device=raw.get("device", "cuda"))
    utils, entrypoint = bridge._import_official_modules()
    outputs = produce_fullwindow_codes(
        mode=str(raw["mode"]),
        code_bank=code_bank,
        targets=targets,
        entrypoint=entrypoint,
        official_utils=utils,
        device=str(raw.get("device", "cuda")),
        memory_kwargs=raw.get("memory"),
        producer_artifacts=(producer_artifacts := {}),
    )
    output_dir = Path(raw["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    producer_state_ref = None
    state_key = "memory_mapper_state" if "memory_mapper_state" in producer_artifacts else "ridge_state" if "ridge_state" in producer_artifacts else None
    if state_key is not None:
        state_path = output_dir / f"{state_key}.pt"
        producer_state_ref = _commit_torch_artifact(
            state_path,
            {
                "schema_version": 1,
                "mode": str(raw["mode"]),
                "source_code_bank": asdict(bank_ref),
                "official_git_commit": official.get("git_commit"),
                "resolved_memory_kwargs": raw.get("memory", {}),
                "state": producer_artifacts[state_key],
            },
            {"kind": state_key, "source_code_bank": asdict(bank_ref)},
        )
    counts: dict[str, int] = {}
    artifact_rows: list[dict[str, Any]] = []
    for payload in outputs:
        payload["producer"]["source_code_bank"] = asdict(bank_ref)
        if producer_state_ref is not None:
            payload["producer"]["fitted_prior_state"] = asdict(producer_state_ref)
        dataset_id = payload["dataset_id"]
        index = counts.get(dataset_id, 0)
        counts[dataset_id] = index + 1
        path = output_dir / dataset_id / f"candidate-{index:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        code_ref = _commit_torch_artifact(
            path,
            payload,
            {"producer": payload["producer"], "code_bank": asdict(bank_ref)},
        )
        artifact_rows.append(
            {
                "dataset": dataset_id,
                "candidate_index": index,
                "prompt_set_sha256": payload["producer"]["prompt_set_sha256"],
                "code_tensor_sha256": payload["producer"]["code_tensor_sha256"],
                "artifact": asdict(code_ref),
            }
        )
        print(f"[weightclip-fullwindow-producer] dataset={dataset_id} mode={raw['mode']} candidate={index} output={path}")
    uniqueness: dict[str, dict[str, float | int]] = {}
    for dataset_id, candidate_count in counts.items():
        tensor_hashes = [
            str(row["code_tensor_sha256"])
            for row in artifact_rows
            if str(row["dataset"]) == dataset_id
        ]
        unique_count = len(set(tensor_hashes))
        uniqueness[dataset_id] = {
            "candidate_count": candidate_count,
            "effective_unique_code_count": unique_count,
            "code_duplicate_rate": 1.0 - unique_count / max(candidate_count, 1),
        }
    write_json_immutable(
        output_dir / "producer_index.json",
        {
            "schema_version": 1,
            "kind": "weightclip_commonzoo_fullwindow_prior_index",
            "mode": str(raw["mode"]),
            "source_code_bank": asdict(bank_ref),
            "fitted_prior_state": None if producer_state_ref is None else asdict(producer_state_ref),
            "code_uniqueness_by_dataset": uniqueness,
            "artifacts": artifact_rows,
        },
    )


def _commit_torch_artifact(path: Path, payload: Mapping[str, Any], ledger: Mapping[str, Any]) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    proposed = ArtifactRef.create(temporary)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    proposed_manifest = {"schema_version": 1, "artifact": {**asdict(proposed), "path": str(path)}, **dict(ledger)}
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            temporary.unlink(missing_ok=True)
            raise FileExistsError(f"partial immutable artifact exists: {path}")
        existing = ArtifactRef.create(path)
        if existing.sha256 != proposed.sha256 or existing.bytes != proposed.bytes:
            temporary.unlink(missing_ok=True)
            raise FileExistsError(f"immutable artifact exists with different bytes: {path}")
        write_json_immutable(manifest_path, proposed_manifest)
        temporary.unlink(missing_ok=True)
        return existing
    temporary.replace(path)
    reference = ArtifactRef.create(path)
    proposed_manifest["artifact"] = asdict(reference)
    write_json_immutable(manifest_path, proposed_manifest)
    return reference


if __name__ == "__main__":
    main()
