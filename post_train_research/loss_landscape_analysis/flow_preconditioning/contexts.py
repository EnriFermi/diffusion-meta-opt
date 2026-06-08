from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .config import ExperimentConfig
from .probe_geometry import ResidualOnlyProbe, ResidualThetaProbe, fit_probe_scales
from .problems import (
    Decoder2DProblem,
    GraphProbe,
    NonsmoothResidualProblem,
    ScalarLandscapeProblem,
    StartItem,
    fit_graph_probe_scales,
)
from .toy_mlp import (
    MLPRegressionProblem,
    QuadraticProblem,
    collect_mlp_flow_pool,
    collect_mlp_heldout_geometry_pool,
    collect_quadratic_flow_pool,
    collect_quadratic_heldout_geometry_pool,
    seed_offset,
)


LossFn = Callable[[torch.Tensor], torch.Tensor]
BatchLossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class ConditionContext:
    experiment: str
    task: str
    condition_name: str
    condition_value: float | str
    seed: int
    dim: int
    probe: Callable[[torch.Tensor], torch.Tensor]
    flow_pool: torch.Tensor
    heldout_pool: torch.Tensor
    tune_items: list[StartItem]
    eval_items: list[StartItem]
    train_loss_factory: Callable[[StartItem], LossFn]
    test_loss_factory: Callable[[StartItem], LossFn]
    flow_steps: int
    train_batch_loss_factory: Callable[[list[StartItem]], BatchLossFn] | None = None
    test_batch_loss_factory: Callable[[list[StartItem]], BatchLossFn] | None = None

    @property
    def numeric_condition(self) -> float:
        try:
            return float(self.condition_value)
        except Exception:
            return 0.0


def _items_from_starts(starts: torch.Tensor) -> list[StartItem]:
    return [StartItem(start=starts[idx]) for idx in range(int(starts.shape[0]))]


def _enabled_experiments(cfg: ExperimentConfig) -> set[str]:
    return {str(value).strip().upper() for value in cfg.experiments}


def build_contexts_for_seed(cfg: ExperimentConfig, *, seed: int, device: torch.device, dtype: torch.dtype) -> list[ConditionContext]:
    contexts: list[ConditionContext] = []
    enabled = _enabled_experiments(cfg)

    if "E0" in enabled:
        e0 = QuadraticProblem(int(cfg.sanity_dim), float(cfg.sanity_condition_number), int(seed), device, dtype)
        contexts.append(
            ConditionContext(
                experiment="E0",
                task="quadratic_residual",
                condition_name="cond_A",
                condition_value=float(cfg.sanity_condition_number),
                seed=int(seed),
                dim=int(cfg.sanity_dim),
                probe=ResidualOnlyProbe(e0.residual),
                flow_pool=collect_quadratic_flow_pool(e0, cfg, seed=int(seed)),
                heldout_pool=collect_quadratic_heldout_geometry_pool(e0, cfg, seed=int(seed)),
                tune_items=_items_from_starts(e0.sample_starts(int(cfg.k_tune), seed=seed_offset(seed, 1001), init_std=float(cfg.init_std))),
                eval_items=_items_from_starts(e0.sample_starts(int(cfg.k_eval), seed=seed_offset(seed, 1002), init_std=float(cfg.init_std))),
                train_loss_factory=lambda _item, problem=e0: problem.train_loss,
                test_loss_factory=lambda _item, problem=e0: problem.test_loss,
                flow_steps=int(cfg.sanity_flow_steps),
            )
        )

    if "E1" in enabled:
        e1 = Decoder2DProblem(cfg, int(seed), device, dtype)
        contexts.append(
            ConditionContext(
                experiment="E1",
                task="fixed_decoder_inverse",
                condition_name="decoder",
                condition_value="fixed",
                seed=int(seed),
                dim=2,
                probe=e1.probe,
                flow_pool=e1.flow_pool(cfg, seed=int(seed)),
                heldout_pool=e1.heldout_pool(cfg, seed=int(seed)),
                tune_items=e1.sample_pairs(int(cfg.k_tune), seed=seed_offset(seed, 1101)),
                eval_items=e1.sample_pairs(int(cfg.k_eval), seed=seed_offset(seed, 1102)),
                train_loss_factory=lambda item, problem=e1: (lambda z, target=item.target: problem.loss_for_target(z, target)),
                test_loss_factory=lambda item, problem=e1: (lambda z, target=item.target: problem.loss_for_target(z, target)),
                flow_steps=int(cfg.flow_steps),
                train_batch_loss_factory=lambda items, problem=e1: (
                    lambda z, targets=torch.stack([item.target for item in items], dim=0): problem.loss_for_targets(z, targets)
                ),
                test_batch_loss_factory=lambda items, problem=e1: (
                    lambda z, targets=torch.stack([item.target for item in items], dim=0): problem.loss_for_targets(z, targets)
                ),
            )
        )

    if "E2" in enabled:
        for objective_name in cfg.e2_objectives:
            for dim in cfg.e2_dims:
                e2 = ScalarLandscapeProblem(cfg, int(seed), int(dim), str(objective_name), device, dtype)
                e2_pool = e2.flow_pool(cfg, seed=int(seed))
                scales = fit_graph_probe_scales(e2.loss, e2_pool)
                for gamma in cfg.e2_gamma_values:
                    contexts.append(
                        ConditionContext(
                            experiment="E2",
                            task=f"{objective_name}_d{dim}",
                            condition_name="gamma",
                            condition_value=float(gamma),
                            seed=int(seed),
                            dim=int(dim),
                            probe=GraphProbe(e2.loss, scales=scales, gamma=float(gamma), rho=1.0, eps=float(cfg.aulc_eps)),
                            flow_pool=e2_pool,
                            heldout_pool=e2.heldout_pool(cfg, seed=seed_offset(seed, 1200 + int(dim))),
                            tune_items=_items_from_starts(e2.sample_starts(int(cfg.k_tune), seed=seed_offset(seed, 1201 + int(dim)))),
                            eval_items=_items_from_starts(e2.sample_starts(int(cfg.k_eval), seed=seed_offset(seed, 1202 + int(dim)))),
                            train_loss_factory=lambda _item, problem=e2: problem.loss,
                            test_loss_factory=lambda _item, problem=e2: problem.loss,
                            flow_steps=int(cfg.flow_steps),
                        )
                    )

    if "E3" in enabled:
        for cond in cfg.e3_condition_numbers:
            e3 = NonsmoothResidualProblem(cfg, int(seed), float(cond), device, dtype)
            contexts.append(
                ConditionContext(
                    experiment="E3",
                    task="relu_residual",
                    condition_name="cond_S",
                    condition_value=float(cond),
                    seed=int(seed),
                    dim=int(cfg.e3_dim),
                    probe=e3.probe,
                    flow_pool=e3.flow_pool(cfg, seed=int(seed)),
                    heldout_pool=e3.heldout_pool(cfg, seed=int(seed)),
                    tune_items=_items_from_starts(e3.sample_starts(int(cfg.k_tune), seed=seed_offset(seed, 1301 + int(cond)))),
                    eval_items=_items_from_starts(e3.sample_starts(int(cfg.k_eval), seed=seed_offset(seed, 1302 + int(cond)))),
                    train_loss_factory=lambda _item, problem=e3: problem.loss,
                    test_loss_factory=lambda _item, problem=e3: problem.loss,
                    flow_steps=int(cfg.flow_steps),
                )
            )

    if "E4" in enabled:
        e4 = MLPRegressionProblem(cfg=cfg, seed=int(seed), device=device, dtype=dtype)
        e4_pool = collect_mlp_flow_pool(e4, cfg, seed=int(seed))
        e4_heldout = collect_mlp_heldout_geometry_pool(e4, cfg, seed=int(seed))
        e4_scales = fit_probe_scales(e4.probe_residual, e4_pool)
        e4_tune = _items_from_starts(e4.sample_starts(int(cfg.k_tune), seed=seed_offset(seed, 1401)))
        e4_eval = _items_from_starts(e4.sample_starts(int(cfg.k_eval), seed=seed_offset(seed, 1402)))
        e4_flow_steps = int(cfg.e4_flow_steps) if int(cfg.e4_flow_steps) > 0 else int(cfg.flow_steps)
        for rho in cfg.e4_rho_values:
            contexts.append(
                ConditionContext(
                    experiment="E4",
                    task="tiny_mlp_weight_space",
                    condition_name="rho",
                    condition_value=float(rho),
                    seed=int(seed),
                    dim=25,
                    probe=ResidualThetaProbe(e4.probe_residual, scales=e4_scales, rho=float(rho), eps=float(cfg.aulc_eps)),
                    flow_pool=e4_pool,
                    heldout_pool=e4_heldout,
                    tune_items=e4_tune,
                    eval_items=e4_eval,
                    train_loss_factory=lambda _item, problem=e4: problem.train_loss,
                    test_loss_factory=lambda _item, problem=e4: problem.test_loss,
                    flow_steps=e4_flow_steps,
                )
            )
    return contexts


def warped_grid_rows(ctx: ConditionContext, trained_flow: torch.nn.Module, random_flow: torch.nn.Module, cfg: ExperimentConfig) -> list[dict[str, object]]:
    if int(ctx.dim) != 2 or ctx.experiment not in {"E1", "E2"}:
        return []
    if ctx.experiment == "E1":
        z0_min, z0_max = float(cfg.e1_domain_z1[0]), float(cfg.e1_domain_z1[1])
        z1_min, z1_max = float(cfg.e1_domain_z2[0]), float(cfg.e1_domain_z2[1])
    else:
        z0_min = z1_min = -4.0
        z0_max = z1_max = 4.0
    rows: list[dict[str, object]] = []
    line_count = 9
    point_count = int(cfg.low_dim_grid_points)
    z0_values = torch.linspace(z0_min, z0_max, point_count, device=ctx.flow_pool.device, dtype=ctx.flow_pool.dtype)
    z1_values = torch.linspace(z1_min, z1_max, point_count, device=ctx.flow_pool.device, dtype=ctx.flow_pool.dtype)
    line_values_0 = torch.linspace(z0_min, z0_max, line_count, device=ctx.flow_pool.device, dtype=ctx.flow_pool.dtype)
    line_values_1 = torch.linspace(z1_min, z1_max, line_count, device=ctx.flow_pool.device, dtype=ctx.flow_pool.dtype)
    with torch.no_grad():
        for flow_name, flow in {"trained_flow": trained_flow, "random_flow": random_flow}.items():
            for line_idx, z0 in enumerate(line_values_0):
                points = torch.stack([torch.full_like(z1_values, z0), z1_values], dim=-1)
                rows.extend(_grid_line_rows(ctx, flow, flow_name, "z0_const", line_idx, points))
            for line_idx, z1 in enumerate(line_values_1):
                points = torch.stack([z0_values, torch.full_like(z0_values, z1)], dim=-1)
                rows.extend(_grid_line_rows(ctx, flow, flow_name, "z1_const", line_idx, points))
    return rows


def _grid_line_rows(
    ctx: ConditionContext,
    flow: torch.nn.Module,
    flow_name: str,
    axis: str,
    line_idx: int,
    points: torch.Tensor,
) -> list[dict[str, object]]:
    mapped = flow(points)[0]
    return [
        {
            "experiment": ctx.experiment,
            "task": ctx.task,
            "condition_name": ctx.condition_name,
            "condition_value": str(ctx.condition_value),
            "seed": int(ctx.seed),
            "flow": flow_name,
            "axis": axis,
            "line_index": int(line_idx),
            "point_index": int(point_idx),
            "z0": float(points[point_idx, 0].cpu().item()),
            "z1": float(points[point_idx, 1].cpu().item()),
            "u0": float(mapped[point_idx, 0].cpu().item()),
            "u1": float(mapped[point_idx, 1].cpu().item()),
        }
        for point_idx in range(int(points.shape[0]))
    ]
