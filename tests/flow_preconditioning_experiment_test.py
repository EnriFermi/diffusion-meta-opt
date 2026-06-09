from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from post_train_research.big_vae_latent_flattening.flow import (
    RQSplineFlow,
    RQSplineFlowConfig,
    RealNVPFlow,
    flow_architecture_sanity,
)
from post_train_research.loss_landscape_analysis.flow_preconditioning import ExperimentConfig, run_or_load
from post_train_research.loss_landscape_analysis.flow_preconditioning.config import torch_dtype
from post_train_research.loss_landscape_analysis.flow_preconditioning.flow_experiment import make_flow
from post_train_research.loss_landscape_analysis.flow_preconditioning.optimization import (
    Curve,
    aulc,
    curve_metrics,
    run_direct_curve,
    run_direct_curves_batched,
    run_direct_curves_batched_loss,
    run_flow_curve,
    run_flow_curves_batched,
    run_flow_curves_batched_loss,
)
from post_train_research.loss_landscape_analysis.flow_preconditioning.probe_geometry import (
    ResidualOnlyProbe,
    ResidualThetaProbe,
    fit_probe_scales,
    metric_tensors_from_jacobians,
    probe_jacobians_for_theta,
)
from post_train_research.loss_landscape_analysis.flow_preconditioning.problems import Decoder2DProblem
from post_train_research.loss_landscape_analysis.flow_preconditioning.contexts import build_contexts_for_seed
from post_train_research.loss_landscape_analysis.flow_preconditioning.e4_debug import (
    E4DebugConfig,
    build_e4_debug_state,
    preconditioner_alignment_rows,
    run_probe_metric_curve,
    run_probe_metric_curves,
    train_e4_debug_flow,
    trajectory_step_alignment_rows,
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

    assert tuple(cfg.experiments) == ("E0", "E1", "E2", "E3", "E4")
    assert cfg.flow_architecture == "rq_spline"
    assert set(cfg.e2_dims) >= {2, 4}
    assert {"rastrigin_abs", "rosenbrock_abs"}.issubset(set(cfg.e2_objectives))
    assert tuple(cfg.e2_gamma_values) == (0.3, 1.0, 3.0)
    assert tuple(cfg.e3_condition_numbers) == (1.0, 100.0, 10000.0)
    assert tuple(cfg.e4_rho_values) == (0.0, 1e-3, 1e-2, 5e-2)


def test_make_flow_defaults_to_rq_spline_and_keeps_realnvp_baseline() -> None:
    rq_flow = make_flow(5, _small_cfg(flow_num_layers=2, flow_hidden_dim=8, flow_network_depth=1))
    realnvp_flow = make_flow(
        5,
        _small_cfg(flow_architecture="realnvp", flow_num_layers=2, flow_hidden_dim=8, flow_network_depth=1),
    )

    assert isinstance(rq_flow, RQSplineFlow)
    assert isinstance(realnvp_flow, RealNVPFlow)


def test_rq_spline_flow_identity_roundtrip_logdet_and_inverse_autograd() -> None:
    flow = RQSplineFlow(
        RQSplineFlowConfig(
            dim=5,
            num_layers=3,
            hidden_dim=8,
            network_depth=1,
            num_bins=8,
            bound=5.0,
        )
    )
    x = torch.randn(7, 5, dtype=torch.float32).clamp(-2.0, 2.0)

    u, log_det = flow(x)
    x_reconstructed, inverse_log_det = flow.inverse(u)
    metrics = flow_architecture_sanity(flow, x, compute_condition=True, condition_samples=3)

    assert torch.allclose(u, x, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_reconstructed, x, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_det, torch.zeros_like(log_det), atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-5, rtol=1e-5)
    assert metrics["roundtrip_max_abs_error"] < 1e-5
    assert metrics["median_abs_displacement"] < 1e-5
    assert metrics["median_abs_logdet"] < 1e-5
    assert metrics["median_condition"] < 1.01

    u_req = u.detach().clone().requires_grad_(True)
    x_inv, inv_log_det = flow.inverse(u_req)
    loss = x_inv.square().sum() + inv_log_det.square().sum()
    loss.backward()
    assert u_req.grad is not None
    assert torch.isfinite(u_req.grad).all()


def test_rq_spline_flow_analytic_inverse_after_small_conditioner_perturbation() -> None:
    torch.manual_seed(123)
    flow = RQSplineFlow(
        RQSplineFlowConfig(
            dim=5,
            num_layers=3,
            hidden_dim=8,
            network_depth=1,
            num_bins=8,
            bound=5.0,
        )
    )
    with torch.no_grad():
        for layer in flow.layers:
            layer.final_linear.weight.normal_(0.0, 1e-3)
            layer.final_linear.bias.normal_(0.0, 1e-3)

    x = (torch.randn(9, 5, dtype=torch.float32) * 2.0).clamp(-4.5, 4.5)
    u, log_det = flow(x)
    x_reconstructed, inverse_log_det = flow.inverse(u)

    assert torch.isfinite(u).all()
    assert torch.isfinite(log_det).all()
    assert torch.allclose(x_reconstructed, x, atol=3e-5, rtol=3e-5)
    assert torch.allclose(log_det + inverse_log_det, torch.zeros_like(log_det), atol=3e-5, rtol=3e-5)


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


def test_e4_only_context_filter_and_flow_steps() -> None:
    cfg = _small_cfg(experiments=("E4",), e4_flow_steps=7)
    contexts = build_contexts_for_seed(cfg, seed=0, device=torch.device("cpu"), dtype=torch.float32)

    assert len(contexts) == len(cfg.e4_rho_values)
    assert {ctx.experiment for ctx in contexts} == {"E4"}
    assert {ctx.flow_steps for ctx in contexts} == {7}


def test_batched_downstream_curves_match_single_curve_runners() -> None:
    cfg = _small_cfg()
    starts = torch.tensor([[1.0, 2.0], [-1.5, 0.25], [0.1, -0.3]], dtype=torch.float32)
    lrs = torch.tensor([1e-2, 3e-3, 1e-3], dtype=torch.float32)
    target = torch.tensor([0.5, -1.0], dtype=torch.float32)

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * (theta - target).square().sum()

    flow = make_flow(2, cfg).to(dtype=torch.float32)

    for optimizer_name in ("sgd", "adam"):
        direct_single = [
            run_direct_curve(
                train_loss_fn=loss_fn,
                test_loss_fn=loss_fn,
                theta0=start,
                optimizer_name=optimizer_name,
                lr=float(lr),
                steps=4,
            )
            for start, lr in zip(starts, lrs, strict=True)
        ]
        direct_batched = run_direct_curves_batched(
            train_loss_fn=loss_fn,
            test_loss_fn=loss_fn,
            theta0_batch=starts,
            optimizer_name=optimizer_name,
            lrs=lrs,
            steps=4,
        )
        flow_single = [
            run_flow_curve(
                train_loss_fn=loss_fn,
                test_loss_fn=loss_fn,
                flow=flow,
                theta0=start,
                optimizer_name=optimizer_name,
                lr=float(lr),
                steps=4,
            )
            for start, lr in zip(starts, lrs, strict=True)
        ]
        flow_batched = run_flow_curves_batched(
            train_loss_fn=loss_fn,
            test_loss_fn=loss_fn,
            flow=flow,
            theta0_batch=starts,
            optimizer_name=optimizer_name,
            lrs=lrs,
            steps=4,
        )

        for single, batched in zip(direct_single, direct_batched, strict=True):
            assert np.allclose(single.train_loss, batched.train_loss)
            assert np.allclose(single.test_loss, batched.test_loss)
            assert np.allclose(single.final_theta, batched.final_theta)
        for single, batched in zip(flow_single, flow_batched, strict=True):
            assert np.allclose(single.train_loss, batched.train_loss)
            assert np.allclose(single.test_loss, batched.test_loss)
            assert np.allclose(single.final_theta, batched.final_theta)


def test_target_batched_downstream_curves_match_single_curve_runners() -> None:
    cfg = _small_cfg()
    problem = Decoder2DProblem(cfg, seed=0, device=torch.device("cpu"), dtype=torch.float32)
    pairs = problem.sample_pairs(3, seed=123)
    starts = torch.stack([item.start for item in pairs], dim=0)
    targets = torch.stack([item.target for item in pairs], dim=0)
    lrs = torch.tensor([1e-2, 3e-3, 1e-3], dtype=torch.float32)
    flow = make_flow(2, cfg).to(dtype=torch.float32)

    def batch_loss(theta_batch: torch.Tensor) -> torch.Tensor:
        return problem.loss_for_targets(theta_batch, targets)

    for optimizer_name in ("sgd", "adam"):
        direct_single = [
            run_direct_curve(
                train_loss_fn=lambda theta, target=item.target: problem.loss_for_target(theta, target),
                test_loss_fn=lambda theta, target=item.target: problem.loss_for_target(theta, target),
                theta0=item.start,
                optimizer_name=optimizer_name,
                lr=float(lr),
                steps=4,
            )
            for item, lr in zip(pairs, lrs, strict=True)
        ]
        direct_batched = run_direct_curves_batched_loss(
            train_loss_batch_fn=batch_loss,
            test_loss_batch_fn=batch_loss,
            theta0_batch=starts,
            optimizer_name=optimizer_name,
            lrs=lrs,
            steps=4,
        )
        flow_single = [
            run_flow_curve(
                train_loss_fn=lambda theta, target=item.target: problem.loss_for_target(theta, target),
                test_loss_fn=lambda theta, target=item.target: problem.loss_for_target(theta, target),
                flow=flow,
                theta0=item.start,
                optimizer_name=optimizer_name,
                lr=float(lr),
                steps=4,
            )
            for item, lr in zip(pairs, lrs, strict=True)
        ]
        flow_batched = run_flow_curves_batched_loss(
            train_loss_batch_fn=batch_loss,
            test_loss_batch_fn=batch_loss,
            flow=flow,
            theta0_batch=starts,
            optimizer_name=optimizer_name,
            lrs=lrs,
            steps=4,
        )

        for single, batched in zip(direct_single, direct_batched, strict=True):
            assert np.allclose(single.train_loss, batched.train_loss)
            assert np.allclose(single.test_loss, batched.test_loss)
            assert np.allclose(single.final_theta, batched.final_theta)
        for single, batched in zip(flow_single, flow_batched, strict=True):
            assert np.allclose(single.train_loss, batched.train_loss)
            assert np.allclose(single.test_loss, batched.test_loss)
            assert np.allclose(single.final_theta, batched.final_theta)


def test_flow_preconditioning_smoke_writes_outputs(tmp_path) -> None:
    cfg = _small_cfg(
        run_label="smoke",
        artifact_root=str(tmp_path),
        downstream_tuning_batch_size=4,
        downstream_eval_batch_size=2,
        parallel_contexts=2,
    )

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


def test_e4_geometry_debug_smoke_writes_outputs(tmp_path) -> None:
    debug_cfg = E4DebugConfig(
        run_label="e4_debug_smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        seed=0,
        rho=1e-2,
        flow_steps=1,
        eval_every=1,
        flow_batch_size=1,
        flow_num_layers=2,
        flow_hidden_dim=8,
        flow_network_depth=1,
        flow_random_samples=2,
        flow_trajectory_count=1,
        flow_trajectory_steps=1,
        heldout_geometry_samples=2,
        train_eval_samples=1,
        heldout_eval_samples=1,
        train_points=8,
        probe_points=8,
        test_points=16,
    )

    state = build_e4_debug_state(debug_cfg)
    result = train_e4_debug_flow(state)

    assert list(result.history["step"]) == [0, 1]
    assert set(result.final_geometry["split"]) == {"train_eval", "heldout_eval"}
    assert set(result.final_geometry["coordinate"]) == {"original", "flow"}
    assert (result.output_dir / "history.csv").is_file()
    assert (result.output_dir / "final_geometry.csv").is_file()
    assert (result.output_dir / "flow_state.pt").is_file()


def test_probe_metric_curves_smoke(tmp_path) -> None:
    debug_cfg = E4DebugConfig(
        run_label="probe_metric_smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        seed=0,
        flow_random_samples=2,
        flow_trajectory_count=1,
        flow_trajectory_steps=1,
        heldout_geometry_samples=2,
        train_eval_samples=1,
        heldout_eval_samples=1,
        train_points=8,
        probe_points=8,
        test_points=16,
    )
    state = build_e4_debug_state(debug_cfg)
    starts = state.problem.sample_starts(2, seed=321)

    curves = run_probe_metric_curves(
        train_loss_fn=state.problem.train_loss,
        test_loss_fn=state.problem.test_loss,
        probe=state.probe,
        theta0_batch=starts,
        lr=1e-2,
        steps=2,
        damping=1e-3,
    )

    assert len(curves) == 2
    for curve in curves:
        assert curve.train_loss.shape == (3,)
        assert curve.test_loss.shape == (3,)
        assert curve.final_theta is not None
        assert curve.final_theta.shape == (25,)
        assert np.isfinite(curve.train_loss).all()


def test_preconditioner_alignment_rows_smoke(tmp_path) -> None:
    debug_cfg = E4DebugConfig(
        run_label="preconditioner_alignment_smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        seed=0,
        flow_steps=1,
        eval_every=1,
        flow_batch_size=1,
        flow_num_layers=2,
        flow_hidden_dim=8,
        flow_network_depth=1,
        flow_random_samples=2,
        flow_trajectory_count=1,
        flow_trajectory_steps=1,
        heldout_geometry_samples=2,
        train_eval_samples=1,
        heldout_eval_samples=1,
        train_points=8,
        probe_points=8,
        test_points=16,
    )
    state = build_e4_debug_state(debug_cfg)
    result = train_e4_debug_flow(state)
    theta_points = state.problem.sample_starts(2, seed=654)

    rows = preconditioner_alignment_rows(
        train_loss_fn=state.problem.train_loss,
        probe=state.probe,
        flow=result.flow,
        theta_points=theta_points,
        damping=1e-3,
        source="test_path",
        start_index=0,
        steps=[0, 1],
    )

    assert len(rows) == 2
    for row in rows:
        assert row["source"] == "test_path"
        assert np.isfinite(float(row["p_metric_norm"]))
        assert np.isfinite(float(row["p_nf_norm"]))
        assert np.isfinite(float(row["norm_ratio_metric_to_nf"]))
        assert -1.0001 <= float(row["cos_p_nf_p_metric"]) <= 1.0001
        assert -1.0001 <= float(row["cos_g_p_metric"]) <= 1.0001
        assert -1.0001 <= float(row["cos_g_p_nf"]) <= 1.0001


def test_probe_metric_trajectory_step_alignment_is_self_consistent(tmp_path) -> None:
    debug_cfg = E4DebugConfig(
        run_label="probe_metric_step_alignment_smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        seed=0,
        flow_steps=1,
        eval_every=1,
        flow_batch_size=1,
        flow_num_layers=2,
        flow_hidden_dim=8,
        flow_network_depth=1,
        flow_random_samples=2,
        flow_trajectory_count=1,
        flow_trajectory_steps=1,
        heldout_geometry_samples=2,
        train_eval_samples=1,
        heldout_eval_samples=1,
        train_points=8,
        probe_points=8,
        test_points=16,
    )
    state = build_e4_debug_state(debug_cfg)
    result = train_e4_debug_flow(state)
    theta0 = state.problem.sample_starts(1, seed=987)[0]
    curve = run_probe_metric_curve(
        train_loss_fn=state.problem.train_loss,
        test_loss_fn=state.problem.test_loss,
        probe=state.probe,
        theta0=theta0,
        lr=1e-2,
        steps=2,
        damping=1e-3,
        store_path=True,
    )
    theta_path = torch.as_tensor(curve.path, dtype=torch.float32)

    rows = trajectory_step_alignment_rows(
        train_loss_fn=state.problem.train_loss,
        probe=state.probe,
        flow=result.flow,
        theta_path=theta_path,
        damping=1e-3,
        source="probe_metric_path",
        start_index=0,
        steps=[0, 1, 2],
    )

    assert len(rows) == 2
    assert min(float(row["cos_step_p_metric"]) for row in rows) > 0.999
