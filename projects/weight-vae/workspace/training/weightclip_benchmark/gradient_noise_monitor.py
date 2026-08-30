from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from big_vae.datasets.operator_bank import BalancedOperatorBankMixer, operator_bank_data_pipeline
from training.big_vae.presliced import _fetch_presliced_training_batch_cpu
from training.weightclip_benchmark.analyze_ae_gradient_batch_stability import (
    _extract_gradient_summary,
    _make_panel_plans,
    _objective_loss,
    _parameter_group,
)


_COMPONENTS = ("structural_direction", "structural_scale_x10")


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


class GradientNoiseMonitor:
    """Periodic, optimizer-free repeatability diagnostic for effective B128.

    Four fixed, disjoint B128 blocks are evaluated at each checkpoint step.
    Their six pairwise cosines directly measure whether independent B128
    gradients agree; no larger batch is treated as ground truth.  The monitor
    owns a separate operator-bank iterator and restores all process RNG state.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        cfg: Any,
        device: torch.device,
        logger: Any,
        csv_path: Path,
        sample_manifest_path: Path,
        every_steps: int,
        start_logical_index: int,
        panel_size: int = 65_536,
        panel_seeds: tuple[int, ...] = (31_001, 31_013, 31_019, 31_033),
    ) -> None:
        if int(cfg.train.slice_batch_size) != 32 or int(cfg.train.grad_accum_steps) != 4:
            raise ValueError("gradient-noise monitor requires physical B32 and grad_accum_steps=4")
        if float(cfg.train.behavioral_coef) != 0.0 or float(cfg.train.structural_coef) != 1.0:
            raise ValueError("gradient-noise monitor requires structural-only training")
        structural = cfg.train.struct_loss
        expected = (
            float(structural.lambda_dir),
            float(structural.lambda_scale),
            float(structural.lambda_rec),
            float(structural.lambda_rel),
        )
        if expected[0] != 1.0 or expected[1] < 0.0 or expected[2:] != (0.0, 0.0):
            raise ValueError(f"unexpected structural loss contract: {expected}")
        big = cfg.model.big_vae
        dropout_values = (
            float(big.dropout),
            float(big.distribution_encoder.dropout),
            float(big.patch_tokenizer.residual_dropout),
        )
        if any(value != 0.0 for value in dropout_values) or bool(big.use_latent_sampling):
            raise ValueError("gradient-noise monitor requires deterministic dropout-free AE")

        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.csv_path = Path(csv_path)
        self.sample_manifest_path = Path(sample_manifest_path)
        self.every_steps = int(every_steps)
        self.start_logical_index = int(start_logical_index)
        self.panel_size = int(panel_size)
        self.panel_seeds = tuple(int(seed) for seed in panel_seeds)
        self.structural_scale_weight = expected[1]
        if self.every_steps <= 0:
            raise ValueError("gradient-noise monitor every_steps must be positive")

        parameter_model = model.module if hasattr(model, "module") else model
        if hasattr(parameter_model, "_orig_mod"):
            parameter_model = parameter_model._orig_mod
        self.named_parameters = {
            name: parameter for name, parameter in parameter_model.named_parameters() if parameter.requires_grad
        }
        grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
        for name, parameter in self.named_parameters.items():
            grouped[_parameter_group(name)].append((name, parameter))
        self.grouped_parameters = dict(grouped)
        self.leaf_groups = sorted(
            group for group in self.grouped_parameters if group.startswith("encoder_")
        )
        if not self.leaf_groups:
            raise RuntimeError("gradient-noise monitor found no encoder parameter groups")
        self.panel_plans, self.group_numel = _make_panel_plans(
            self.grouped_parameters,
            panel_size=self.panel_size,
            seeds=list(self.panel_seeds),
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_cpu_state = torch.get_rng_state()
        torch_cuda_state = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        try:
            self.probe_blocks = self._load_probe_blocks()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state(torch_cuda_state, self.device)
        self._write_sample_manifest()

    def _load_probe_blocks(self) -> list[list[Any]]:
        operator = self.cfg.train.operator_bank
        batches: list[Any] = []
        with operator_bank_data_pipeline(
            str(operator.pair_manifest),
            seed=int(self.cfg.data.seed),
            repeat=True,
            permutation_views=bool(operator.permutation_views),
            canonical_probability=float(operator.canonical_probability),
            hot_shards=int(operator.hot_shards),
            expected_pair_manifest_sha256=str(operator.pair_manifest_sha256),
            rank=0,
            world_size=1,
            max_active_strata=int(operator.max_active_strata),
            max_active_bundle_bytes=int(operator.max_active_bundle_bytes),
            logger=self.logger,
        ) as (dataset, sampler):
            sampler.set_start_index(self.start_logical_index)
            mixer = BalancedOperatorBankMixer(
                dataset,
                (dataset[request] for request in sampler),
                start_index=self.start_logical_index,
            )
            for _ in range(16):
                batches.append(
                    _fetch_presliced_training_batch_cpu(
                        dataset_iter=mixer,
                        batch_size=32,
                        logger=self.logger,
                    )
                )
        observed = tuple(index for batch in batches for index in batch.logical_indices)
        expected = tuple(range(self.start_logical_index, self.start_logical_index + 512))
        if observed != expected:
            raise RuntimeError("gradient-noise B128 probe window is not exact and contiguous")
        return [batches[offset : offset + 4] for offset in range(0, 16, 4)]

    def _write_sample_manifest(self) -> None:
        self.sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "weightclip_periodic_b128_gradient_repeatability_samples_v2",
            "effective_batch_size": 128,
            "structural_direction_weight": 1.0,
            "structural_scale_weight": self.structural_scale_weight,
            "independent_blocks": 4,
            "pairwise_comparisons": 6,
            "microbatch_size": 32,
            "blocks": [
                {
                    "block": block_index,
                    "logical_indices": [
                        int(index) for batch in block for index in batch.logical_indices
                    ],
                }
                for block_index, block in enumerate(self.probe_blocks)
            ],
        }
        self.sample_manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    def should_run(self, step: int) -> bool:
        return int(step) > 0 and int(step) % self.every_steps == 0

    def _leaf_panel(self, panels: dict[tuple[str, int], np.ndarray], group: str, seed: int) -> np.ndarray:
        vector = panels[(group, seed)].astype(np.float64, copy=False)
        return vector * math.sqrt(float(self.group_numel[group]) / max(1, int(vector.size)))

    def _group_panel(
        self,
        panels: dict[tuple[str, int], np.ndarray],
        group: str,
        seed: int,
    ) -> np.ndarray:
        if group != "encoder_total":
            return self._leaf_panel(panels, group, seed)
        return np.concatenate([self._leaf_panel(panels, leaf, seed) for leaf in self.leaf_groups])

    def _backward_block(self, objective: str, block: list[Any]) -> tuple[float, dict[str, Any], dict[Any, np.ndarray]]:
        self.model.zero_grad(set_to_none=True)
        loss_value = 0.0
        for batch in block:
            tensors = [
                tensor.to(self.device, non_blocking=False)
                for tensor in (batch.W, batch.x, batch.x_mask, batch.d_in_mask, batch.d_out_mask)
            ]
            W, x, x_mask, d_in_mask, d_out_mask = tensors
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast:
                loss = _objective_loss(
                    objective,
                    model=self.model,
                    cfg=self.cfg,
                    W=W,
                    x=x,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                )
                scaled = loss / 4.0
            scaled.backward()
            loss_value += float(loss.detach().item()) / 4.0
            del W, x, x_mask, d_in_mask, d_out_mask, loss, scaled
        exact, panels = _extract_gradient_summary(
            self.named_parameters,
            self.grouped_parameters,
            self.panel_plans,
            self.group_numel,
            panel_size=self.panel_size,
            seeds=list(self.panel_seeds),
        )
        return loss_value, exact, panels

    def _append_rows(self, rows: list[dict[str, Any]]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.csv_path.exists()
        fields = list(rows[0])
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def run(self, *, step: int) -> dict[str, float]:
        started = time.perf_counter()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_cpu_state = torch.get_rng_state()
        torch_cuda_state = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        was_training = bool(self.model.training)
        component_records: dict[str, list[dict[str, Any]]] = {}
        try:
            self.model.train()
            for component_index, objective in enumerate(_COMPONENTS):
                random.seed(91_000 + component_index)
                np.random.seed(91_000 + component_index)
                torch.manual_seed(91_000 + component_index)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(91_000 + component_index)
                records: list[dict[str, Any]] = []
                for block_index, block in enumerate(self.probe_blocks):
                    loss, exact, panels = self._backward_block(objective, block)
                    records.append(
                        {"block": block_index, "loss": loss, "exact": exact, "panels": panels}
                    )
                component_records[objective] = records

            objectives = (*_COMPONENTS, "structural")
            groups = (*self.leaf_groups, "encoder_total")
            rows: list[dict[str, Any]] = []
            comet_metrics: dict[str, float] = {}
            for objective in objectives:
                display_objective = (
                    "structural_scale_weighted"
                    if objective == "structural_scale_x10"
                    else objective
                )
                for group in groups:
                    per_seed_metrics: list[dict[str, float]] = []
                    panel_errors: list[float] = []
                    for seed in self.panel_seeds:
                        block_vectors: list[np.ndarray] = []
                        for block_index in range(len(self.probe_blocks)):
                            if objective == "structural":
                                vector = sum(
                                    self._group_panel(
                                        component_records[component][block_index]["panels"],
                                        group,
                                        seed,
                                    )
                                    for component in _COMPONENTS
                                )
                            else:
                                vector = self._group_panel(
                                    component_records[objective][block_index]["panels"],
                                    group,
                                    seed,
                                )
                            block_vectors.append(vector)
                        pairwise_cosines: list[float] = []
                        pairwise_relative_differences: list[float] = []
                        positive = 0
                        for left_index, right_index in combinations(range(len(block_vectors)), 2):
                            left = block_vectors[left_index]
                            right = block_vectors[right_index]
                            cosine = _cosine(left, right)
                            pairwise_cosines.append(cosine)
                            denominator = 0.5 * (
                                float(np.linalg.norm(left)) + float(np.linalg.norm(right))
                            )
                            pairwise_relative_differences.append(
                                float(np.linalg.norm(left - right) / denominator)
                                if denominator > 0.0
                                else float("nan")
                            )
                            positive += int(float(np.dot(left, right)) > 0.0)
                        norms = [float(np.linalg.norm(vector)) for vector in block_vectors]
                        mean_norm = statistics.fmean(norms)
                        norm_cv = (
                            float(statistics.pstdev(norms) / mean_norm)
                            if mean_norm > 0.0
                            else float("nan")
                        )
                        mean_vector = np.mean(np.stack(block_vectors), axis=0)
                        per_seed_metrics.append(
                            {
                                "median_pairwise_cosine": _median(pairwise_cosines),
                                "p10_pairwise_cosine": _quantile(pairwise_cosines, 0.10),
                                "min_pairwise_cosine": min(pairwise_cosines),
                                "positive_pair_fraction": positive / len(pairwise_cosines),
                                "median_pairwise_relative_difference": _median(
                                    pairwise_relative_differences
                                ),
                                "p90_pairwise_relative_difference": _quantile(
                                    pairwise_relative_differences, 0.90
                                ),
                                "gradient_norm_cv": norm_cv,
                                "mean_resultant_ratio": (
                                    float(np.linalg.norm(mean_vector)) / mean_norm
                                    if mean_norm > 0.0
                                    else float("nan")
                                ),
                            }
                        )
                    metrics = {
                        key: _median([seed_metrics[key] for seed_metrics in per_seed_metrics])
                        for key in per_seed_metrics[0]
                    }
                    if (
                        metrics["median_pairwise_cosine"] < 0.10
                        or metrics["positive_pair_fraction"] <= 0.50
                    ):
                        state = "noise_dominated"
                    elif metrics["median_pairwise_cosine"] < 0.50:
                        state = "high_variance"
                    else:
                        state = "directionally_consistent"
                    if objective in _COMPONENTS and group != "encoder_total":
                        for block_index, record in enumerate(component_records[objective]):
                            exact_l2 = float(record["exact"][group]["exact_l2"])
                            panel_l2_values = [
                                float(np.linalg.norm(self._group_panel(record["panels"], group, seed)))
                                for seed in self.panel_seeds
                            ]
                            if exact_l2 > 0.0:
                                panel_errors.append(abs(_median(panel_l2_values) - exact_l2) / exact_l2)
                    row = {
                        "step": int(step),
                        "objective": display_objective,
                        "objective_weight": (
                            self.structural_scale_weight
                            if objective == "structural_scale_x10"
                            else 1.0
                        ),
                        "group": group,
                        "effective_batch_size": 128,
                        "independent_blocks": len(self.probe_blocks),
                        "pairwise_comparisons": math.comb(len(self.probe_blocks), 2),
                        **metrics,
                        "panel_l2_error_median": _median(panel_errors),
                        "state": state,
                        "elapsed_seconds": float(time.perf_counter() - started),
                    }
                    rows.append(row)
                    prefix = f"gradient_noise/{display_objective}/{group}"
                    comet_metrics[f"{prefix}/median_pairwise_cosine"] = float(
                        metrics["median_pairwise_cosine"]
                    )
                    comet_metrics[f"{prefix}/p10_pairwise_cosine"] = float(
                        metrics["p10_pairwise_cosine"]
                    )
                    comet_metrics[f"{prefix}/positive_pair_fraction"] = float(
                        metrics["positive_pair_fraction"]
                    )
                    comet_metrics[f"{prefix}/median_pairwise_relative_difference"] = float(
                        metrics["median_pairwise_relative_difference"]
                    )
                    comet_metrics[f"{prefix}/gradient_norm_cv"] = float(metrics["gradient_norm_cv"])
                    comet_metrics[f"{prefix}/mean_resultant_ratio"] = float(
                        metrics["mean_resultant_ratio"]
                    )
                    comet_metrics[f"{prefix}/state"] = {
                        "noise_dominated": 0.0,
                        "high_variance": 1.0,
                        "directionally_consistent": 2.0,
                    }[state]
            self._append_rows(rows)
            comet_metrics["gradient_noise/elapsed_seconds"] = float(time.perf_counter() - started)
            self.logger.info(
                "gradient_noise_monitor step=%s elapsed=%.1fs encoder_total=%s",
                step,
                comet_metrics["gradient_noise/elapsed_seconds"],
                {
                    row["objective"]: {
                        "median_pairwise_cos": row["median_pairwise_cosine"],
                        "p10_pairwise_cos": row["p10_pairwise_cosine"],
                        "norm_cv": row["gradient_norm_cv"],
                        "state": row["state"],
                    }
                    for row in rows
                    if row["group"] == "encoder_total"
                },
            )
            return comet_metrics
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state(torch_cuda_state, self.device)


__all__ = ["GradientNoiseMonitor"]
