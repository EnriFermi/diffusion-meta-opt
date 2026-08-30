#!/usr/bin/env python3
"""Resolve the Stage-D/G DAG solely from immutable upstream artifact indices."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

from big_vae.weightclip_benchmark.manifests import write_json_atomic
from training.weightclip_benchmark.prepare_stage_dg_configs import _write_yaml_immutable, resolve_configs


def _command(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(item) for item in args)]


def _stage(
    name: str,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path] | None = None,
) -> dict[str, Any]:
    missing = [str(path) for path in inputs if not path.is_file()]
    return {
        "stage": name,
        "status": "ready" if not missing else "blocked_upstream",
        "missing": missing,
        "inputs": [str(path) for path in inputs],
        "produces": [str(path) for path in outputs or []],
        "command": command,
    }


def assert_dag_reachability(dag: Mapping[str, Any]) -> None:
    stages = list(dag["stages"])
    names = [str(stage["stage"]) for stage in stages]
    required = {
        *(f"validate_decoded_e4_{name}" for name in (
            "ours_anchor_free", "ours_oracle_anchor",
            "weightclip_anchor_free", "weightclip_oracle_anchor",
        )),
        "select_global_flow_e4",
        *(f"seal_flow_{name}" for name in (
            "ours_anchor_free", "ours_oracle_anchor",
            "weightclip_anchor_free", "weightclip_oracle_anchor",
        )),
        "build_production_candidate_spec",
    }
    if not required <= set(names):
        raise ValueError(f"Stage-D/G DAG omits required E4/seal nodes: {sorted(required - set(names))}")
    produced_at: dict[str, int] = {}
    external_inputs = set(map(str, dag.get("external_inputs", [])))
    for index, stage in enumerate(stages):
        for output in stage.get("produces", []):
            if output in produced_at:
                raise ValueError(f"DAG output has multiple producers: {output}")
            produced_at[str(output)] = index
    for index, stage in enumerate(stages):
        for input_path in stage.get("inputs", []):
            producer = produced_at.get(str(input_path))
            if producer is None and str(input_path) not in external_inputs:
                raise ValueError(f"DAG consumer {stage['stage']} has dangling generated input {input_path}")
            if producer is not None and producer >= index:
                raise ValueError(f"DAG consumer {stage['stage']} precedes producer for {input_path}")
    for seal in map(str, dag["flow_seals"]):
        if seal not in produced_at:
            raise ValueError(f"flow seal has no DAG producer: {seal}")
    candidate_stage = stages[names.index("build_production_candidate_spec")]
    missing_seal_edges = set(map(str, dag["flow_seals"])) - set(map(str, candidate_stage.get("inputs", [])))
    if missing_seal_edges:
        raise ValueError(f"candidate producer is not gated on all flow seals: {sorted(missing_seal_edges)}")
    serialized = json.dumps(dag, sort_keys=True)
    if any(token in serialized for token in ("REQUIRED_", "PENDING", "CONTENT_ADDRESSED", "ONE_OF_")):
        raise ValueError("resolved DAG contains an unresolved placeholder")


def resolve_dag(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {key: Path(value).resolve() for key, value in config["paths"].items()}
    root = paths["artifact_root"]
    resolved_dir = root / "stage_dg_configs"
    resolution_id = "stage_dg_v3_final_three_seed"
    dataset_index = resolve_configs(
        zoo_config=paths["zoo_config"], evaluation_config=paths["evaluation_config"], output_dir=resolved_dir
    )
    evaluation_resolved_path = Path(dataset_index["evaluation_config"]["path"])
    resolved_evaluation = yaml.safe_load(evaluation_resolved_path.read_text())
    official = resolved_evaluation["official_weightclip"]
    evaluation_seeds = [int(seed) for seed in resolved_evaluation["evaluation"]["evaluation_seeds"]]
    checkpoint_manifest = paths["checkpoint_manifest"]
    bank_state_path = paths["operator_bank_root"] / "build_state.json"
    pair_candidates = sorted(glob.glob(str(paths["operator_bank_root"] / "operator_dataset-*.json")))
    if len(pair_candidates) > 1:
        raise ValueError(f"ambiguous operator pair manifests: {pair_candidates}")
    pair_manifest: Path | None = None
    if bank_state_path.is_file():
        bank_state = json.loads(bank_state_path.read_text(encoding="utf-8"))
        pair_manifest = Path(str(bank_state["pair_manifest"])).resolve()
        if not pair_manifest.is_file() or str(pair_manifest) not in pair_candidates:
            raise ValueError("operator bank build_state does not bind the unique content-addressed pair manifest")
    ae_seal = paths["ae_seal"]
    ours_bundle_dir = root / "task_bundles" / "ours"
    wc_bundle_dir = root / "task_bundles" / "weightclip"
    source_datasets = dataset_index["source_datasets"]
    common_bundle = {
        "device": "cuda",
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_index_zero_based": 44,
        "datasets": source_datasets,
        "context_indices": list(range(512)),
        "prompt_indices": list(range(10)),
        "official_weightclip": {
            "repo_path": official["repo_path"], "cache_dir": official["cache_dir"],
            "checkpoint_path": official["checkpoint_path"],
            "dataset_encoder_path": official["dataset_encoder_path"],
        },
    }
    ours_bundle_config = resolved_dir / f"source_bundles_ours.{resolution_id}.yaml"
    wc_bundle_config = resolved_dir / f"source_bundles_weightclip.{resolution_id}.yaml"
    if pair_manifest is not None:
        _write_yaml_immutable(
            ours_bundle_config,
            common_bundle | {
                "codec": "ours", "output_dir": str(ours_bundle_dir),
                "codec_seal": str(ae_seal), "pair_manifest": str(pair_manifest), "encode_batch_size": 8,
            },
        )
    _write_yaml_immutable(
        wc_bundle_config,
        common_bundle | {"codec": "weightclip", "output_dir": str(wc_bundle_dir)},
    )
    parity_seal = root / "task_fit_parity_seal.json"
    ours_fit_config = resolved_dir / f"task_latent_fit_ours.{resolution_id}.yaml"
    wc_fit_config = resolved_dir / f"task_latent_fit_weightclip.{resolution_id}.yaml"
    ours_bundle_index = ours_bundle_dir / "bundle_index.json"
    wc_bundle_index = wc_bundle_dir / "bundle_index.json"
    ours_records = root / "task_latents" / "ours"
    wc_records = root / "task_latents" / "weightclip"
    wc_fit_raw = yaml.safe_load(paths["weightclip_fit_config"].read_text(encoding="utf-8"))
    wc_fit_raw["fit"]["output_dir"] = str(wc_records)
    _write_yaml_immutable(wc_fit_config, wc_fit_raw)
    if ae_seal.is_file():
        seal = json.loads(ae_seal.read_text(encoding="utf-8"))
        ours_fit_raw = yaml.safe_load(paths["ours_fit_config"].read_text(encoding="utf-8"))
        ours_fit_raw["decoder"]["checkpoint"] = seal["checkpoint"]["path"]
        ours_fit_raw["decoder"]["kwargs"]["model_config_path"] = seal["model_config"]["path"]
        ours_fit_raw["fit"]["output_dir"] = str(ours_records)
        _write_yaml_immutable(ours_fit_config, ours_fit_raw)
    inventory = root / "task_fit_inventory.json"
    validation_inventory = root / "e4_validation_inventory.json"
    ours_fit_completion = root / "task_latents" / "ours.fit_completion.json"
    wc_fit_completion = root / "task_latents" / "weightclip.fit_completion.json"
    generated_configs: list[Path] = [
        evaluation_resolved_path,
        ours_bundle_config,
        wc_bundle_config,
        ours_fit_config,
        wc_fit_config,
    ]
    stages = [
        _stage(
            "seal_task_fit_parity",
            _command(
                "training.weightclip_benchmark.seal_task_fit_parity",
                "--ours-config", ours_fit_config, "--weightclip-config", wc_fit_config, "--output", parity_seal,
            ),
            [ours_fit_config, wc_fit_config],
            [parity_seal],
        ),
        _stage(
            "build_source_bundles_ours",
            _command("training.weightclip_benchmark.build_task_fit_bundles", "--config", ours_bundle_config),
            [checkpoint_manifest, bank_state_path, ours_bundle_config, ae_seal],
            [ours_bundle_index],
        ),
        _stage(
            "build_source_bundles_weightclip",
            _command("training.weightclip_benchmark.build_task_fit_bundles", "--config", wc_bundle_config),
            [checkpoint_manifest, wc_bundle_config],
            [wc_bundle_index],
        ),
        _stage(
            "fit_all_ours",
            _command(
                "training.weightclip_benchmark.fit_all_task_latents",
                "--bundle-index", ours_bundle_index, "--fit-config", ours_fit_config,
                "--parity-seal", parity_seal,
                "--completion-output", ours_fit_completion,
            ),
            [ours_bundle_index, ours_fit_config, parity_seal],
            [ours_fit_completion],
        ),
        _stage(
            "fit_all_weightclip",
            _command(
                "training.weightclip_benchmark.fit_all_task_latents",
                "--bundle-index", wc_bundle_index, "--fit-config", wc_fit_config,
                "--parity-seal", parity_seal,
                "--completion-output", wc_fit_completion,
            ),
            [wc_bundle_index, wc_fit_config, parity_seal],
            [wc_fit_completion],
        ),
        _stage(
            "seal_task_fit_inventory",
            _command(
                "training.weightclip_benchmark.build_task_fit_inventory",
                "--ours-bundle-index", ours_bundle_index,
                "--weightclip-bundle-index", wc_bundle_index,
                "--ours-record-glob", ours_records / "*.pt",
                "--weightclip-record-glob", wc_records / "*.pt",
                "--parity-seal", parity_seal,
                "--ours-completion", ours_fit_completion,
                "--weightclip-completion", wc_fit_completion,
                "--validation-output", validation_inventory,
                "--output", inventory,
            ),
            [ours_bundle_index, wc_bundle_index, parity_seal, ours_fit_completion, wc_fit_completion],
            [inventory, validation_inventory],
        ),
    ]
    # Flow configs are fully resolved to the one inventory path. The training
    # command becomes ready only after that immutable inventory exists.
    flow_seals: list[Path] = []
    flow_infos: list[dict[str, Any]] = []
    for base_name in (
        "flow_ours_anchor_free.yaml", "flow_ours_oracle_anchor.yaml",
        "flow_weightclip_anchor_free.yaml", "flow_weightclip_oracle_anchor.yaml",
    ):
        raw = yaml.safe_load((paths["config_root"] / base_name).read_text(encoding="utf-8"))
        codec = str(raw["data"]["codec"])
        raw["data"]["record_globs"] = [str((ours_records if codec == "ours" else wc_records) / "*.pt")]
        raw["data"]["matched_record_globs"] = {
            "ours": [str(ours_records / "*.pt")], "weightclip": [str(wc_records / "*.pt")]
        }
        raw["data"]["expected_inventory_path"] = str(inventory)
        raw["train"]["output_dir"] = str(root / "flows" / Path(base_name).stem.removeprefix("flow_"))
        resolved_flow = resolved_dir / f"{Path(base_name).stem}.{resolution_id}.yaml"
        _write_yaml_immutable(resolved_flow, raw)
        generated_configs.append(resolved_flow)
        flow_seal = Path(raw["train"]["output_dir"]) / "flow_seal.json"
        flow_seals.append(flow_seal)
        flow_infos.append(
            {
                "name": Path(base_name).stem.removeprefix("flow_"),
                "codec": codec,
                "path_kind": str(raw["train"]["path_kind"]),
                "output_dir": Path(raw["train"]["output_dir"]),
                "seal": flow_seal,
            }
        )
        stages.append(
            _stage(
                f"train_{Path(base_name).stem}",
                _command("training.weightclip_benchmark.train_flow", "--config", resolved_flow),
                [inventory, resolved_flow],
                [Path(raw["train"]["output_dir"]) / "latest.pt", Path(raw["train"]["output_dir"]) / "latent_normalizer.pt"],
            )
        )
    validation_payload = (
        json.loads(validation_inventory.read_text(encoding="utf-8")) if validation_inventory.is_file() else None
    )
    e4_reports: list[Path] = []
    for info in flow_infos:
        name = str(info["name"])
        codec = str(info["codec"])
        output_dir = Path(info["output_dir"])
        checkpoint = output_dir / "latest.pt"
        normalizer = output_dir / "latent_normalizer.pt"
        report = output_dir / "decoded_e4_validation.json"
        e4_reports.append(report)
        e4_config = resolved_dir / f"decoded_e4_{name}.{resolution_id}.yaml"
        generated_configs.append(e4_config)
        if validation_payload is not None:
            decoder: dict[str, Any]
            if codec == "ours":
                decoder = {"codec_seal": str(ae_seal)}
            else:
                decoder = {
                    "factory": (
                        "big_vae.weightclip_benchmark.decoder_adapters:"
                        "build_official_weightclip_tokenizer_only_decoder_adapter"
                    ),
                    "checkpoint": official["checkpoint_path"],
                    "kwargs": {
                        "official_repo": official["repo_path"],
                        "cache_dir": official["cache_dir"],
                        "dataset_encoder_path": official["dataset_encoder_path"],
                        "window_size": 512,
                    },
                }
            _write_yaml_immutable(
                e4_config,
                {
                    "codec": codec,
                    "device": "cuda",
                    "checkpoint": str(checkpoint),
                    "normalizer": str(normalizer),
                    "validation_record_globs": [
                        row["path"] for row in validation_payload["records"][codec]
                    ],
                    "task_bundle_globs": [
                        row["path"] for row in validation_payload["task_bundles"][codec]
                    ],
                    "expected_validation_inventory": str(validation_inventory),
                    "decoder": decoder,
                    "output_path": str(report),
                },
            )
        stages.append(
            _stage(
                f"validate_decoded_e4_{name}",
                _command("training.weightclip_benchmark.validate_decoded_flow", "--config", e4_config),
                [validation_inventory, e4_config, checkpoint, normalizer, *([ae_seal] if codec == "ours" else [])],
                [report],
            )
        )
    global_e4 = root / "flows" / "global_e4_selection.json"
    select_command = _command("training.weightclip_benchmark.select_flow_e4")
    for report in e4_reports:
        select_command.extend(["--report", str(report)])
    select_command.extend(["--output", str(global_e4)])
    stages.append(_stage("select_global_flow_e4", select_command, e4_reports, [global_e4]))
    for info, report in zip(flow_infos, e4_reports, strict=True):
        output_dir = Path(info["output_dir"])
        stages.append(
            _stage(
                f"seal_flow_{info['name']}",
                _command(
                    "training.weightclip_benchmark.seal_flow",
                    "--checkpoint", output_dir / "latest.pt",
                    "--normalizer", output_dir / "latent_normalizer.pt",
                    "--decoded-validation", report,
                    "--global-e4-selection", global_e4,
                    "--codec", info["codec"],
                    "--output", info["seal"],
                ),
                [output_dir / "latest.pt", output_dir / "latent_normalizer.pt", report, global_e4],
                [Path(info["seal"])],
            )
        )
    ood_config = Path(dataset_index["ood_conditioning_config"]["path"])
    anchor_config = Path(dataset_index["ood_anchor_config"]["path"])
    generated_configs.extend([ood_config, anchor_config])
    ood_bundle_cfg = yaml.safe_load(ood_config.read_text(encoding="utf-8"))
    conditioning_root = root / "ood_conditioning_bundles"
    conditioning_paths = [conditioning_root / f"{dataset}.pt" for dataset in sorted(ood_bundle_cfg["datasets"])]
    anchor_manifest = root / "ood_anchors" / "anchor_manifest.jsonl"
    stages.extend(
        [
            _stage(
                "build_ood_conditioning",
                _command("training.weightclip_benchmark.build_ood_conditioning_bundles", "--config", ood_config),
                [ood_config],
                conditioning_paths,
            ),
            _stage(
                "train_ood_anchors",
                _command("training.weightclip_benchmark.train_ood_anchors", "--config", anchor_config),
                [anchor_config],
                [anchor_manifest],
            ),
        ]
    )
    anchor_codec_config = resolved_dir / f"ood_anchor_codec_bundles.{resolution_id}.yaml"
    _write_yaml_immutable(
        anchor_codec_config,
        {
            "device": "cuda", "anchor_manifest": str(anchor_manifest),
            "ours_codec_seal": str(ae_seal),
            "output_dir": str(root / "ood_anchor_codec_bundles"),
            "official_weightclip": ood_bundle_cfg["official_weightclip"],
            "datasets": ood_bundle_cfg["datasets"],
            "encode_batch_size": 8, "capture_batch_size": 32,
        },
    )
    generated_configs.append(anchor_codec_config)
    anchor_codec_index = root / "ood_anchor_codec_bundles" / "bundle_index.json"
    stages.append(
        _stage(
            "build_ood_anchor_codec_bundles",
            _command("training.weightclip_benchmark.build_ood_anchor_codec_bundles", "--config", anchor_codec_config),
            [anchor_manifest, ae_seal, anchor_codec_config],
            [anchor_codec_index],
        )
    )
    fullwindow_bank = root / "weightclip_fullwindow_bank.pt"
    fullwindow_bank_config = resolved_dir / f"weightclip_fullwindow_bank.{resolution_id}.yaml"
    _write_yaml_immutable(
        fullwindow_bank_config,
        {"source_bundle_index": str(wc_bundle_index), "output_path": str(fullwindow_bank)},
    )
    generated_configs.append(fullwindow_bank_config)
    stages.append(
        _stage(
            "build_weightclip_fullwindow_bank",
            _command("training.weightclip_benchmark.build_weightclip_fullwindow_bank", "--config", fullwindow_bank_config),
            [wc_bundle_index, fullwindow_bank_config],
            [fullwindow_bank],
        )
    )
    code_root = root / "weightclip_codes"
    code_indexes: list[Path] = []
    for mode in (
        "ridge", "memory", "nearest_code", "ridge_native_oracle",
        "memory_native_oracle", "nearest_code_native_oracle",
    ):
        output_dir = code_root / mode
        producer_config = resolved_dir / f"weightclip_codes_{mode}_microbatch1.{resolution_id}.yaml"
        producer_payload: dict[str, Any] = {
            "device": "cuda",
            "mode": mode,
            "code_bank": str(fullwindow_bank),
            "conditioning_bundles": [str(path) for path in conditioning_paths],
            "official_weightclip": {
                "repo_path": official["repo_path"], "cache_dir": official["cache_dir"],
                "git_commit": official["git_commit"],
            },
            "output_dir": str(output_dir),
        }
        if mode.startswith("memory"):
            producer_payload["memory"] = {
                "seed": 0, "temperature": 0.5, "top_k": 32, "epochs": 200,
                "lr": 0.00005, "batch_size": 1, "d_model": 512,
                "n_head": 8, "n_layer": 4, "leave_one_out": True,
                "use_amp": True, "token_subsample": None, "patience": 300,
                "log_every": 25,
            }
        _write_yaml_immutable(producer_config, producer_payload)
        generated_configs.append(producer_config)
        producer_index = output_dir / "producer_index.json"
        code_indexes.append(producer_index)
        stages.append(
            _stage(
                f"produce_weightclip_{mode}_codes",
                _command(
                    "training.weightclip_benchmark.build_weightclip_multiwindow_codes",
                    "--config", producer_config,
                ),
                [fullwindow_bank, producer_config, *conditioning_paths],
                [producer_index],
            )
        )
    expected_grid = root / "expected_evaluation_grid.final_three_seed.json"
    candidate_spec = resolved_dir / f"candidate_spec.{resolution_id}.yaml"
    candidate_spec_builder_config = resolved_dir / f"candidate_spec_builder.{resolution_id}.yaml"
    _write_yaml_immutable(
        candidate_spec_builder_config,
        {
            "benchmark_tier": "final",
            "datasets": sorted(ood_bundle_cfg["datasets"]),
            "num_classes": {
                dataset: int(row["num_classes"]) for dataset, row in ood_bundle_cfg["datasets"].items()
            },
            "evaluation_seeds": evaluation_seeds,
            "ae_seal": str(ae_seal),
            "flow_seals": {
                "ours_anchor_free": str(flow_seals[0]),
                "ours_oracle_anchor": str(flow_seals[1]),
                "weightclip_anchor_free": str(flow_seals[2]),
                "weightclip_oracle_anchor": str(flow_seals[3]),
            },
            "conditioning_bundle_root": str(conditioning_root),
            "anchor_manifest": str(anchor_manifest),
            "anchor_codec_bundle_index": str(anchor_codec_index),
            "weightclip_code_root": str(code_root),
            "official_weightclip": {
                "checkpoint_path": official["checkpoint_path"],
                "dataset_encoder_path": official["dataset_encoder_path"],
            },
            "expected_grid_output": str(expected_grid),
            "candidate_spec_output": str(candidate_spec),
        },
    )
    generated_configs.append(candidate_spec_builder_config)
    exploratory_evaluation = resolved_dir / f"evaluation.exploratory_seed0.{resolution_id}.yaml"
    exploratory_evaluation_payload = yaml.safe_load(evaluation_resolved_path.read_text(encoding="utf-8"))
    exploratory_evaluation_payload["experiment_id"] = "weightclip_resnet18slim_v1_exploratory_seed0"
    exploratory_evaluation_payload["evaluation"]["evaluation_seeds"] = [0]
    exploratory_evaluation_payload["scientific_status"] = (
        "exploratory_only_not_valid_for_final_expected_grid_unseal_or_reporting"
    )
    _write_yaml_immutable(exploratory_evaluation, exploratory_evaluation_payload)
    exploratory_grid = root / "expected_evaluation_grid.exploratory_seed0.json"
    exploratory_spec = resolved_dir / f"candidate_spec.exploratory_seed0.{resolution_id}.yaml"
    exploratory_builder = resolved_dir / f"candidate_spec_builder.exploratory_seed0.{resolution_id}.yaml"
    exploratory_builder_payload = yaml.safe_load(candidate_spec_builder_config.read_text(encoding="utf-8"))
    exploratory_builder_payload.update(
        {
            "benchmark_tier": "exploratory_single_seed0",
            "evaluation_seeds": [0],
            "expected_grid_output": str(exploratory_grid),
            "candidate_spec_output": str(exploratory_spec),
        }
    )
    _write_yaml_immutable(exploratory_builder, exploratory_builder_payload)
    stages.append(
        _stage(
            "build_production_candidate_spec",
            _command(
                "training.weightclip_benchmark.build_production_candidate_spec",
                "--config", candidate_spec_builder_config,
            ),
            [
                candidate_spec_builder_config,
                ae_seal,
                anchor_manifest,
                anchor_codec_index,
                *conditioning_paths,
                *flow_seals,
                *code_indexes,
                Path(official["checkpoint_path"]),
                Path(official["dataset_encoder_path"]),
            ],
            [candidate_spec, expected_grid],
        )
    )
    candidate_manifest = root / "candidate_manifest.final_three_seed.jsonl"
    stages.append(
        _stage(
            "build_candidate_manifest",
            _command(
                "training.weightclip_benchmark.build_candidate_manifest",
                "--config", candidate_spec, "--output", candidate_manifest,
            ),
            [candidate_spec, expected_grid],
            [candidate_manifest, candidate_manifest.with_suffix(".index.json")],
        )
    )
    approval_file = root / "ood_test_unseal_approval.json"
    unseal = root / "ood_test_unseal.json"
    unseal_command = _command(
        "training.weightclip_benchmark.unseal_ood_evaluation",
        "--evaluation-config", evaluation_resolved_path,
        "--candidate-manifest", candidate_manifest,
        "--expected-grid", expected_grid,
        "--ae-seal", ae_seal,
    )
    for flow_seal in flow_seals:
        unseal_command.extend(["--flow-seal", str(flow_seal)])
    unseal_command.extend(["--approval-file", str(approval_file), "--output", str(unseal)])
    stages.append(
        _stage(
            "unseal_ood_evaluation",
            unseal_command,
            [evaluation_resolved_path, candidate_manifest, expected_grid, ae_seal, *flow_seals, approval_file],
            [unseal],
        )
    )
    stages.append(
        _stage(
            "evaluate_stage_g",
            _command(
                "training.weightclip_benchmark.evaluate",
                "--config", evaluation_resolved_path,
                "--candidate-manifest", candidate_manifest,
                "--unseal", unseal,
                "--output-dir", root / "evaluation_results",
            ),
            [evaluation_resolved_path, candidate_manifest, unseal],
        )
    )
    dag = {
        "schema_version": 1,
        "kind": "artifact_driven_stage_dg_dag",
        "resolution_id": resolution_id,
        "ae_user_approval_gate": str(ae_seal),
        "stages": stages,
        "flow_seals": [str(path) for path in flow_seals],
        "external_inputs": [
            str(path)
            for path in sorted(
                {
                    checkpoint_manifest,
                    bank_state_path,
                    ae_seal,
                    approval_file,
                    Path(official["checkpoint_path"]),
                    Path(official["dataset_encoder_path"]),
                    *generated_configs,
                },
                key=str,
            )
        ],
        "artifact_driven_rerun": "rerun resolver after each completed stage; readiness derives only from immutable outputs",
        "only_user_gates": {
            "ae_seal": str(ae_seal),
            "ood_test_unseal_approval": str(approval_file),
        },
        "exploratory_seed0": {
            "scientific_status": "exploratory_only_not_valid_for_final_unseal_or_reporting",
            "evaluation_config": str(exploratory_evaluation),
            "candidate_spec_builder_config": str(exploratory_builder),
            "candidate_spec_command": _command(
                "training.weightclip_benchmark.build_production_candidate_spec",
                "--config", exploratory_builder,
            ),
            "candidate_manifest_command": _command(
                "training.weightclip_benchmark.build_candidate_manifest",
                "--config", exploratory_spec,
                "--output", root / "candidate_manifest.exploratory_seed0.jsonl",
            ),
        },
    }
    assert_dag_reachability(dag)
    output = Path(config["dag_output"]).resolve()
    write_json_atomic(output, dag)
    return dag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("conf/weightclip_benchmark/stage_dg_pipeline.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dag = resolve_dag(config)
    print(f"[stage-dg-dag] stages={len(dag['stages'])} output={config['dag_output']}")


if __name__ == "__main__":
    main()
