from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.flow_preconditioning import ExperimentConfig, run_or_load
from post_train_research.loss_landscape_analysis.flow_preconditioning.config import torch_dtype
from post_train_research.loss_landscape_analysis.flow_preconditioning.optimization import Curve, aulc, curve_metrics
from post_train_research.loss_landscape_analysis.flow_preconditioning.probe_geometry import (
    ResidualOnlyProbe,
    ResidualThetaProbe,
    fit_probe_scales,
    metric_tensors_from_jacobians,
    probe_jacobians_for_theta,
)
from post_train_research.loss_landscape_analysis.flow_preconditioning.runner import _select_lrs
from post_train_research.loss_landscape_analysis.flow_preconditioning.toy_mlp import (
    MLP_DIM,
    MLPRegressionProblem,
    mlp_forward,
    pack_theta,
    shifted_grid,
    unpack_theta,
)


def _small_cfg(**overrides) -> ExperimentConfig:
    values = {
        "device": "cpu",
        "seeds": (0,),
        "k_tune": 1,
        "k_eval": 1,
        "budgets": (1,),
        "e2_gamma_values": (1.0,),
        "e2_dims": (2,),
        "e2_objectives": ("rastrigin_abs",),
        "e3_condition_numbers": (1.0,),
        "e4_rho_values": (1e-2,),
        "e4_main_rho": 1e-2,
        "flow_steps": 1,
        "sanity_flow_steps": 1,
        "flow_batch_size": 2,
        "flow_log_every": 1,
        "flow_random_samples": 4,
        "flow_trajectory_count": 1,
        "flow_trajectory_steps": 1,
        "sanity_random_samples": 4,
        "sanity_trajectory_count": 1,
        "sanity_trajectory_steps": 1,
        "heldout_geometry_samples": 2,
        "sanity_heldout_geometry_samples": 2,
        "e1_random_samples": 4,
        "e1_trajectory_count": 1,
        "e1_trajectory_steps": 1,
        "e1_heldout_geometry_samples": 2,
        "e2_random_samples": 4,
        "e2_trajectory_count": 1,
        "e2_trajectory_steps": 1,
        "e2_heldout_geometry_samples": 2,
        "e3_random_samples": 4,
        "e3_trajectory_count": 1,
        "e3_trajectory_steps": 1,
        "e3_heldout_geometry_samples": 2,
        "sgd_lrs": (1e-3,),
        "adam_lrs": (1e-3,),
        "bootstrap_samples": 10,
        "save_figures": False,
        "cache_first": False,
        "force_rerun": True,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_mlp_flat_parameter_roundtrip_and_dimension() -> None:
    theta = torch.arange(MLP_DIM, dtype=torch.float32)
    w1, b1, w2, b2 = unpack_theta(theta)

    packed = pack_theta(w1, b1, w2, b2)

    assert int(theta.numel()) == 25
    assert torch.equal(packed, theta)
    assert tuple(mlp_forward(theta, torch.linspace(-1, 1, 7)).shape) == (7,)


def test_shifted_grids_are_deterministic_non_identical_and_bounded() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    train = shifted_grid(128, 0.0, device=device, dtype=dtype)
    probe = shifted_grid(64, 0.37, device=device, dtype=dtype)
    test = shifted_grid(512, 0.73, device=device, dtype=dtype)
    train_again = shifted_grid(128, 0.0, device=device, dtype=dtype)

    assert torch.equal(train, train_again)
    assert float(train.min()) >= -2.0
    assert float(test.max()) <= 2.0
    assert not torch.allclose(train[:64], probe)
    assert not torch.allclose(train[:128], test[:128])


def test_probe_map_dimensions_and_normalization_scales() -> None:
    cfg = _small_cfg()
    problem = MLPRegressionProblem(cfg, seed=0, device=torch.device("cpu"), dtype=torch_dtype(cfg))
    pool = problem.sample_starts(5, seed=123)

    scales = fit_probe_scales(problem.probe_residual, pool)
    probe = ResidualThetaProbe(problem.probe_residual, scales=scales, rho=1e-2)
    value = probe(pool[0])

    assert scales.residual_rms > 0.0
    assert scales.theta_rms > 0.0
    assert tuple(value.shape) == (int(cfg.probe_points) + MLP_DIM,)


def test_exact_geometry_for_linear_residual_matches_analytic_metric() -> None:
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    probe = ResidualOnlyProbe(lambda theta: matrix @ theta)
    samples = torch.randn(3, 4)

    jacobians = probe_jacobians_for_theta(probe, samples, create_graph=False)
    metric, trace_g, trace_g2 = metric_tensors_from_jacobians(jacobians)

    expected_metric = matrix.t() @ matrix
    assert torch.allclose(metric, expected_metric.expand_as(metric))
    assert torch.allclose(trace_g, torch.full((3,), float(torch.trace(expected_metric))))
    assert torch.allclose(trace_g2, torch.full((3,), float(expected_metric.square().sum())))


def test_lr_selection_uses_tuning_rows_only() -> None:
    tuning = pd.DataFrame(
        [
            {
                "experiment": "mlp",
                "task": "task",
                "condition_name": "rho",
                "condition_value": "0.01",
                "seed": 0,
                "budget": 10,
                "family": "direct",
                "method": "direct_sgd",
                "optimizer": "sgd",
                "lr": 1e-3,
                "aulc": 10.0,
            },
            {
                "experiment": "mlp",
                "task": "task",
                "condition_name": "rho",
                "condition_value": "0.01",
                "seed": 0,
                "budget": 10,
                "family": "direct",
                "method": "direct_sgd",
                "optimizer": "sgd",
                "lr": 1e-2,
                "aulc": 1.0,
            },
        ]
    )

    selected = _select_lrs(tuning)

    assert float(selected.iloc[0]["selected_lr"]) == 1e-2


def test_default_config_contains_required_e0_e4_sweeps() -> None:
    cfg = ExperimentConfig()

    assert set(cfg.e2_dims) >= {2, 4}
    assert {"rastrigin_abs", "rosenbrock_abs"}.issubset(set(cfg.e2_objectives))
    assert tuple(cfg.e2_gamma_values) == (0.3, 1.0, 3.0)
    assert tuple(cfg.e3_condition_numbers) == (1.0, 100.0, 10000.0)
    assert tuple(cfg.e4_rho_values) == (0.0, 1e-3, 1e-2, 5e-2)


def test_aulc_and_curve_metrics() -> None:
    curve = Curve(
        train_loss=np.array([10.0, 2.0, 1.0], dtype=np.float64),
        test_loss=np.array([11.0, 3.0, 1.5], dtype=np.float64),
    )

    value = aulc(curve.train_loss, budget=2, l_ref=1.0, eps=1e-12)
    metrics = curve_metrics(curve, budget=2, l_ref=1.0, eps=1e-12, success_threshold=1.1)

    assert np.isfinite(value)
    assert metrics["final_train_loss"] == 1.0
    assert metrics["best_train_loss"] == 1.0
    assert metrics["success"] == 1.0


def test_flow_preconditioning_smoke_writes_outputs(tmp_path) -> None:
    cfg = _small_cfg(run_label="smoke", artifact_root=str(tmp_path))

    tables = run_or_load(cfg)

    assert not tables.results.empty
    assert not tables.curves.empty
    assert not tables.geometry.empty
    assert not tables.selected_lrs.empty
    assert set(tables.results["experiment"].unique()) == {"E0", "E1", "E2", "E3", "E4"}
    assert "coordinate_map" in set(tables.geometry["diagnostic"].unique())
    assert (tables.output_dir / "results.parquet").is_file()
    assert (tables.output_dir / "results.csv").is_file()
    assert (tables.output_dir / "curves.parquet").is_file()
    assert (tables.output_dir / "geometry.parquet").is_file()
    assert (tables.output_dir / "selected_lrs.csv").is_file()
    assert (tables.output_dir / "config.json").is_file()
