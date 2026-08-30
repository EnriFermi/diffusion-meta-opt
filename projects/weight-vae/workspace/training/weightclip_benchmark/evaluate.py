#!/usr/bin/env python3
"""Manifest-driven common evaluation launcher.

Generators materialize candidate descriptors into JSONL.  Project-specific
callables build models and loaders; this launcher owns the frozen candidate,
head, BatchNorm, fine-tuning, logging, and artifact protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

from big_vae.weightclip_benchmark.contract import CandidateProtocol, DEFAULT_CONTRACT
from big_vae.weightclip_benchmark.coverage import validate_candidate_cardinality, validate_exact_grid
from big_vae.weightclip_benchmark.evaluation import (
    Candidate,
    EvaluationRun,
    FineTuneConfig,
    evaluate_initializations,
    seed_everything,
    validate_candidate_group_contract,
)
from big_vae.weightclip_benchmark.official_bridge import redact_secrets
from big_vae.weightclip_benchmark.latent_bundles import ArtifactRef
from big_vae.weightclip_benchmark.manifests import write_json_immutable
from big_vae.weightclip_benchmark.reporting import write_evaluation_artifacts
from training.weightclip_benchmark.unseal_ood_evaluation import validate_final_grid_payload


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _group_cache_path(output_dir: Path, key: tuple[str, str, int, str]) -> Path:
    method, dataset, seed, protocol = key
    return output_dir / "group_results" / f"{method}__{dataset}__seed-{seed}__{protocol}.json"


def _selection_commit_path(output_dir: Path, key: tuple[str, str, int, str]) -> Path:
    method, dataset, seed, protocol = key
    return output_dir / "selection_commits" / f"{method}__{dataset}__seed-{seed}__{protocol}.json"


def _selection_committer(
    output_dir: Path,
    group_key: tuple[str, str, int, str],
) -> Callable[[dict[str, Any]], None]:
    method, dataset, seed, protocol = group_key

    def commit(payload: dict[str, Any]) -> None:
        write_json_immutable(
            _selection_commit_path(output_dir, group_key),
            payload
            | {
                "method": method,
                "dataset": dataset,
                "evaluation_seed": seed,
                "protocol": protocol,
            },
        )

    return commit


def _load_group_cache(path: Path, *, contract_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("contract_sha256") != contract_sha256:
        raise ValueError(f"cached evaluation group contract drift: {path}")
    if not isinstance(payload.get("rows"), list) or not isinstance(payload.get("summary"), dict):
        raise ValueError(f"cached evaluation group is malformed: {path}")
    return payload["rows"], payload["summary"]


def _commit_group_cache(
    path: Path,
    *,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_json_immutable(
        path,
        {
            "schema_version": 1,
            "contract_sha256": _canonical_sha256(contract),
            "contract": contract,
            "rows": rows,
            "summary": summary,
        },
    )


def _resolve_callable(spec: str) -> Callable[..., Any]:
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError(f"Callable must use module:function syntax, got {spec!r}")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Configured object is not callable: {spec}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Evaluation config must be a mapping")
    return payload


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["candidates"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("Candidate manifest must contain a list of mappings")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--unseal", type=Path, required=True, help="Explicit post-freeze OOD test authorization")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    unseal = json.loads(args.unseal.read_text(encoding="utf-8"))
    if unseal.get("status") != "OOD_TEST_UNLOCKED" or unseal.get("contract_fingerprint") != DEFAULT_CONTRACT.fingerprint():
        raise ValueError("evaluation is blocked: invalid OOD unseal artifact")
    expected_config = ArtifactRef.from_mapping(unseal["evaluation_config"])
    expected_candidates = ArtifactRef.from_mapping(unseal["candidate_manifest"])
    if expected_config.verify("unsealed evaluation config") != args.config.resolve():
        raise ValueError("OOD unseal binds a different evaluation config")
    if expected_candidates.verify("unsealed candidate manifest") != args.candidate_manifest.resolve():
        raise ValueError("OOD unseal binds a different candidate manifest")
    expected_grid_ref = ArtifactRef.from_mapping(unseal["expected_grid"])
    grid_payload = json.loads(expected_grid_ref.verify("unsealed expected grid").read_text(encoding="utf-8"))
    validate_final_grid_payload(grid_payload)
    runtime = dict(config.get("runtime", {}))
    device = str(args.device or runtime.get("device", "cuda"))
    verbose = not args.quiet and bool(runtime.get("verbose", True))
    adapter = dict(config.get("adapter", {}))
    model_builder = _resolve_callable(str(adapter["model_factory"]))
    loaders_builder = _resolve_callable(str(adapter["loaders_factory"]))
    manifest_rows = _load_manifest(args.candidate_manifest)
    validate_candidate_cardinality(manifest_rows)
    validate_exact_grid(manifest_rows, grid_payload.get("cells", grid_payload))
    manifest_hash = _sha256(args.candidate_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "schema_version": 1,
        "kind": "evaluation_run_contract",
        "evaluation_config": {
            "path": str(args.config.resolve()), "sha256": _sha256(args.config), "bytes": args.config.stat().st_size,
        },
        "candidate_manifest": {
            "path": str(args.candidate_manifest.resolve()), "sha256": manifest_hash,
            "bytes": args.candidate_manifest.stat().st_size,
        },
        "unseal": {
            "path": str(args.unseal.resolve()), "sha256": _sha256(args.unseal), "bytes": args.unseal.stat().st_size,
        },
        "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
    }
    write_json_immutable(args.output_dir / "evaluation_run_contract.json", run_contract)
    resolved = redact_secrets(
        {
            "config": config,
            "config_path": str(args.config.resolve()),
            "candidate_manifest": str(args.candidate_manifest.resolve()),
            "candidate_manifest_sha256": manifest_hash,
            "output_dir": str(args.output_dir.resolve()),
            "device": device,
            "python": sys.version,
            "platform": platform.platform(),
            "contract_fingerprint": DEFAULT_CONTRACT.fingerprint(),
        }
    )
    (args.output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[evaluation:startup] device={device} dtype=model_defined seed=manifest cache=adapter_defined "
        f"candidates={len(manifest_rows)} manifest_sha256={manifest_hash} output={args.output_dir.resolve()}"
    )
    print(f"[evaluation:startup] resolved_config={args.output_dir / 'resolved_config.json'}")

    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        required = {"method", "dataset", "evaluation_seed", "protocol", "candidate_id", "payload"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Candidate manifest row missing {sorted(missing)}: {row}")
        key = (str(row["method"]), str(row["dataset"]), int(row["evaluation_seed"]), str(row["protocol"]))
        groups[key].append(row)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    loader_cache: dict[str, dict[str, Any]] = {}
    for group_index, ((method, dataset, evaluation_seed, protocol_name), records) in enumerate(sorted(groups.items()), start=1):
        group_key = (method, dataset, evaluation_seed, protocol_name)
        protocol = CandidateProtocol(protocol_name)
        group_contract = {
            "run_contract_sha256": _canonical_sha256(run_contract),
            "group": [method, dataset, evaluation_seed, protocol_name],
            "candidate_rows_sha256": _canonical_sha256(records),
        }
        group_cache_path = _group_cache_path(args.output_dir, group_key)
        cached_group = _load_group_cache(
            group_cache_path, contract_sha256=_canonical_sha256(group_contract)
        )
        if cached_group is not None:
            cached_rows, cached_summary = cached_group
            all_rows.extend(cached_rows)
            summaries.append(cached_summary)
            print(f"[evaluation:cache] group={group_index}/{len(groups)} hit={group_cache_path}")
            continue
        seed_everything(evaluation_seed)
        if dataset not in loader_cache:
            print(f"[evaluation:stage] stage=data_loading dataset={dataset} cache=miss")
            loaders = loaders_builder(dataset=dataset, config=config)
            required_loaders = {"train", "selection", "test"}
            if not isinstance(loaders, dict) or not required_loaders <= set(loaders):
                raise TypeError(f"loaders_factory must return keys {sorted(required_loaders)}")
            loader_cache[dataset] = loaders
        else:
            print(f"[evaluation:stage] stage=data_loading dataset={dataset} cache=hit")
        loaders = loader_cache[dataset]
        candidates = [
            Candidate(
                candidate_id=str(row["candidate_id"]),
                payload=row["payload"],
                generation_seconds=float(row.get("generation_seconds", 0.0)),
                generation_nfe=int(row.get("generation_nfe", 0)),
                metadata=dict(row.get("metadata", {})),
            )
            for row in records
        ]
        head_policy, batchnorm_policy, head_module_path = validate_candidate_group_contract(method, records)
        run = EvaluationRun(
            experiment_id=str(config.get("experiment_id", "weightclip_benchmark")),
            method=method,
            dataset=dataset,
            evaluation_seed=evaluation_seed,
            protocol=protocol,
            head_policy=head_policy,
            batchnorm_policy=batchnorm_policy,
            head_module_path=head_module_path,
            verbose=verbose,
        )
        fine = FineTuneConfig(**dict(config.get("fine_tuning", {})))
        print(
            f"[evaluation:stage] stage=downstream group={group_index}/{len(groups)} "
            f"method={method} dataset={dataset} seed={evaluation_seed} protocol={protocol.value}"
        )
        output = evaluate_initializations(
            run,
            candidates,
            lambda candidate, _builder=model_builder: _builder(payload=candidate.payload, config=config),
            train_loader=loaders["train"],
            selection_loader=loaders["selection"],
            test_loader=loaders["test"],
            bn_loader=loaders.get("bn", loaders["train"]),
            device=device,
            finetune=fine,
            selection_committer=_selection_committer(args.output_dir, group_key),
        )
        all_rows.extend(output.rows)
        summaries.append(output.summary)
        _commit_group_cache(
            group_cache_path,
            contract=group_contract,
            rows=output.rows,
            summary=output.summary,
        )
        print(f"[evaluation:cache] committed={group_cache_path}")

    provenance = resolved | {"group_summaries": summaries}
    report = write_evaluation_artifacts(
        args.output_dir,
        all_rows,
        provenance=provenance,
        expected_grid=grid_payload.get("cells", grid_payload),
    )
    if not report.valid:
        raise SystemExit("Evaluation artifacts failed validity review; inspect review.json")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
