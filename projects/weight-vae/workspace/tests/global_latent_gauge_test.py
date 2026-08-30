from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    FlatSpec,
    TaskTensorSet,
    WeightNormalizer,
)
from scripts.audit_latent_raw_common_state_rate import AdamState, _adam_delta
from scripts import evaluate_global_latent_gauge_intervention as gauge_harness
from scripts.evaluate_global_latent_gauge_intervention import (
    PRODUCTION_EVAL_SOURCES,
    PRODUCTION_FIT_SOURCES,
    PRODUCTION_LABELS,
    PRODUCTION_STEPS,
    REAL_SMOKE_STEPS,
    Arm,
    ProductionVAEContext,
    _augmented_components,
    _celo_decode_gauged,
    _celo_explicit_fp32_jacobian,
    _celo_literal_start,
    _fit_global_metric,
    _gauge_loss_grad,
    _identity_gauge,
    _partition_protocol_banks,
    _paired_contrast_outputs,
    _pair_trust_gate,
    _primary_aggregation,
    _propagate_execution_flags,
    _real_coupled_local_curves,
    _real_gauge_curve,
    _real_metric_fit,
    _request_hash,
    _task_hashes,
    _validate_coupled_normal_tangents,
    _validate_exact_grid,
    _validate_real_smoke_selection,
    _validate_same_batches,
    _verify_artifact_hashes,
    _whitening_gauge,
    _write_hash_index,
)


class _TinyWeightVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(6, 2, bias=False)
        self.decoder = torch.nn.Linear(2, 6, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.3, 0.0, 0.4, -0.2],
                        [-0.3, 0.5, 0.1, 0.2, -0.2, 0.1],
                    ]
                )
            )
            self.decoder.weight.copy_(
                torch.tensor(
                    [[1.0, 0.2], [-0.4, 0.8], [0.3, -0.5], [0.7, 0.1], [-0.2, 0.6], [0.4, -0.3]]
                )
            )

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.encoder(values)
        return mean, torch.zeros_like(mean)

    def decode_norm(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def _tiny_task_and_spec() -> tuple[TaskTensorSet, FlatSpec]:
    images = torch.tensor([0.0, 0.4, 0.8, 1.0], dtype=torch.float32).reshape(4, 1, 1, 1)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    task = TaskTensorSet(
        task_name="tiny",
        train_images=images,
        train_labels=labels,
        test_images=images.flip(0),
        test_labels=labels.flip(0),
    )
    spec = FlatSpec(
        keys=("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"),
        shapes=((1, 1), (1,), (2, 1), (2,)),
        sizes=(1, 1, 2, 2),
        model_kind="celo_meta_mlp",
        image_shape=(1, 1, 1),
        hidden_dim=1,
        num_classes=2,
    )
    return task, spec


def _tiny_context(tmp_path) -> ProductionVAEContext:
    vae = _TinyWeightVAE().float().eval()
    normalizer = WeightNormalizer(mean=torch.zeros(6), std=torch.ones(6))
    return ProductionVAEContext(
        label="control_seed0",
        run_dir=tmp_path,
        cfg=SimpleNamespace(show_progress=False, progress_backend="text"),
        vae=vae,
        normalizer=normalizer,
        checkpoint_payload={},
        raw_lr=1.0e-3,
        latent_lr=1.0e-2,
        checkpoint_sha256="checkpoint",
        config_sha256="config",
        selected_lrs_sha256="lrs",
        normalizer_sha256="normalizer",
    )


def test_gauge_algebra_preserves_decode_and_transforms_metric() -> None:
    decoder = torch.tensor([[1.0, 2.0], [-0.5, 1.5], [0.25, -0.75]], dtype=torch.float64)
    metric = decoder.T @ decoder
    gauge = _whitening_gauge(metric, kind="full", rcond=1.0e-5)
    z = torch.tensor([0.3, -0.7], dtype=torch.float64)
    y = gauge.to_gauge(z)

    assert torch.allclose(gauge.from_gauge(y), z, atol=1e-12, rtol=1e-12)
    assert torch.allclose(decoder @ gauge.from_gauge(y), decoder @ z, atol=1e-12, rtol=1e-12)
    expected = (decoder @ gauge.inverse).T @ (decoder @ gauge.inverse)
    assert torch.allclose(gauge.metric_to_gauge(metric), expected, atol=1e-12, rtol=1e-12)


def test_orthogonal_gauge_leaves_sgd_decoded_step_invariant() -> None:
    metric = torch.eye(4, dtype=torch.float64)
    identity = _identity_gauge(metric)
    generator = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
    orthogonal = identity.__class__("orthogonal", None, q, q.T, metric, torch.ones(4), torch.ones(4), 0, 0.0)
    z = torch.randn(4, generator=generator, dtype=torch.float64)
    grad = torch.randn(4, generator=generator, dtype=torch.float64)
    lr = 0.07

    identity_step = identity.from_gauge(identity.to_gauge(z) - lr * identity.gradient_to_gauge(grad))
    orthogonal_step = orthogonal.from_gauge(orthogonal.to_gauge(z) - lr * orthogonal.gradient_to_gauge(grad))

    assert torch.allclose(identity_step, orthogonal_step, atol=1e-12, rtol=1e-12)


def test_full_and_diagonal_flooring_use_exact_declared_cutoff() -> None:
    metric = torch.diag(torch.tensor([4.0, 1.0e-10, 1.0], dtype=torch.float64))
    for kind in ("full", "diagonal"):
        gauge = _whitening_gauge(metric, kind=kind, rcond=1.0e-4)
        expected_floor = 1.0e-4 * float(gauge.raw_eigenvalues.max())
        assert float(gauge.floored_eigenvalues.min()) == pytest.approx(expected_floor)
        assert gauge.clipped_count == 1
        assert torch.allclose(gauge.inverse @ gauge.matrix, torch.eye(3, dtype=torch.float64), atol=1e-12)


def test_metric_fit_deduplicates_within_start_then_equal_weights_starts() -> None:
    records = pd.DataFrame(
        [
            {"source_weight_index": 10, "state_sha256": "a", "gram": torch.eye(2, dtype=torch.float64)},
            {"source_weight_index": 10, "state_sha256": "a", "gram": torch.eye(2, dtype=torch.float64)},
            {"source_weight_index": 10, "state_sha256": "b", "gram": 3.0 * torch.eye(2, dtype=torch.float64)},
            {"source_weight_index": 11, "state_sha256": "c", "gram": 10.0 * torch.eye(2, dtype=torch.float64)},
        ]
    )
    metric, kept = _fit_global_metric(records, expected_fit_sources=[10, 11], evaluation_sources=[99])

    assert torch.equal(metric, 6.0 * torch.eye(2, dtype=torch.float64))
    assert len(kept) == 3
    assert int(kept.loc[kept["state_sha256"] == "a", "duplicate_count"].iloc[0]) == 2
    with pytest.raises(ValueError, match="leaked"):
        _fit_global_metric(records, expected_fit_sources=[10, 11], evaluation_sources=[11])


def test_protocol_banks_are_fixed_disjoint_rows() -> None:
    bank = pd.DataFrame(
        {
            "start_bank_position": range(32),
            "source_weight_index": list(PRODUCTION_EVAL_SOURCES) + list(PRODUCTION_FIT_SOURCES),
        }
    )
    evaluation, metric_fit = _partition_protocol_banks(bank, require_exact=False)

    assert evaluation["source_weight_index"].tolist() == list(PRODUCTION_EVAL_SOURCES)
    assert metric_fit["source_weight_index"].tolist() == list(PRODUCTION_FIT_SOURCES)
    assert set(evaluation["source_weight_index"]).isdisjoint(metric_fit["source_weight_index"])


def test_augmented_arms_hold_tangent_equal_and_placebo_is_orthogonal() -> None:
    jacobian = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [0.5, -0.25], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    ambient = torch.tensor([1.0, -2.0, 0.5, 3.0, -4.0], dtype=torch.float32)
    components = _augmented_components(jacobian, ambient, placebo_seed=19)
    projector = components["projector"]
    tangent = torch.as_tensor(components["tangent"])
    task_normal = torch.as_tensor(components["task_normal"])
    placebo = torch.as_tensor(components["placebo"])

    assert torch.allclose(jacobian @ torch.as_tensor(components["latent_preimage"]), tangent, atol=2e-6)
    assert torch.allclose(projector.project(task_normal), torch.zeros_like(task_normal), atol=2e-6)
    assert torch.allclose(projector.project(placebo), torch.zeros_like(placebo), atol=2e-6)
    assert float(torch.dot(task_normal, placebo).abs()) <= 2e-6
    assert placebo.norm() == pytest.approx(task_normal.norm(), rel=2e-6)


def test_exact_grid_rejects_missing_replaced_and_duplicate_rows() -> None:
    dimensions = {"seed": ["s0", "s1"], "arm": ["a", "b"], "step": [0, 1]}
    exact = pd.MultiIndex.from_product(dimensions.values(), names=dimensions.keys()).to_frame(index=False)
    assert _validate_exact_grid(exact, dimensions=dimensions, key_columns=tuple(dimensions), artifact_name="fixture")["accepted"]

    with pytest.raises(ValueError, match="not exact"):
        _validate_exact_grid(exact.iloc[:-1], dimensions=dimensions, key_columns=tuple(dimensions), artifact_name="fixture")
    duplicate = pd.concat([exact, exact.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicates=1"):
        _validate_exact_grid(duplicate, dimensions=dimensions, key_columns=tuple(dimensions), artifact_name="fixture")
    replaced = exact.copy()
    replaced.loc[0, "arm"] = "outside"
    with pytest.raises(ValueError, match="missing=1 unexpected=1"):
        _validate_exact_grid(replaced, dimensions=dimensions, key_columns=tuple(dimensions), artifact_name="fixture")


def test_primary_aggregation_weights_starts_within_seed_then_seeds_equally() -> None:
    rows = []
    starts_by_seed = {PRODUCTION_LABELS[0]: [1, 2], PRODUCTION_LABELS[1]: [3], PRODUCTION_LABELS[2]: [4]}
    post0_loss = {PRODUCTION_LABELS[0]: 0.0, PRODUCTION_LABELS[1]: 10.0, PRODUCTION_LABELS[2]: 20.0}
    for seed, starts in starts_by_seed.items():
        for source in starts:
            for step in range(PRODUCTION_STEPS + 1):
                rows.append(
                    {
                        "vae_seed": seed,
                        "source_weight_index": source,
                        "arm_id": "z_sgd:identity:none",
                        "step": step,
                        "train_loss": 30.0 if step == 0 else post0_loss[seed],
                    }
                )
    _, per_seed, aggregate = _primary_aggregation(pd.DataFrame(rows))

    assert per_seed["vae_seed_mean_progress"].tolist() == pytest.approx([30.0, 20.0, 10.0])
    assert float(aggregate.iloc[0]["equal_weight_vae_seed_mean_progress"]) == pytest.approx(20.0)
    assert int(aggregate.iloc[0]["n_vae_seeds"]) == 3


def test_paired_contrasts_preserve_cutoff_and_equal_seed_hierarchy() -> None:
    arm_ids = {
        "z_sgd:full:1e-6",
        "z_sgd:full:1e-5",
        "z_sgd:full:1e-4",
        "z_sgd:identity:none",
        "z_sgd:orthogonal:none",
        "z_sgd:diagonal:1e-6",
        "z_sgd:diagonal:1e-5",
        "z_sgd:diagonal:1e-4",
        "pullback",
        "augmented_task_normal",
        "augmented_placebo",
    }
    rows = []
    for seed_index, vae_seed in enumerate(PRODUCTION_LABELS):
        for source in PRODUCTION_EVAL_SOURCES:
            for arm_index, arm_id in enumerate(sorted(arm_ids)):
                rows.append(
                    {
                        "vae_seed": vae_seed,
                        "source_weight_index": source,
                        "arm_id": arm_id,
                        "train_post0_aulc_progress": float(seed_index + arm_index),
                        "trajectory_treatment_executable": source != PRODUCTION_EVAL_SOURCES[-1],
                    }
                )
    paired, seeds, equal = _paired_contrast_outputs(pd.DataFrame(rows))

    assert len(paired) == 10 * 3 * 16
    assert len(seeds) == 10 * 2 * 3
    assert len(equal) == 10 * 2
    full_diagonal = paired[paired["contrast"].str.startswith("full_minus_diagonal")]
    assert set(full_diagonal["cutoff"]) == {"1e-6", "1e-5", "1e-4"}
    assert equal.loc[equal["analysis_set"] == "all_start_itt", "n_vae_seeds"].eq(3).all()
    paired_normal = paired[paired["contrast"] == "task_normal_minus_placebo"]
    assert paired_normal["tangent_held_equal_by_design"].astype(bool).all()
    reallocation = paired[paired["contrast"] == "task_normal_minus_pullback"]
    assert not reallocation["tangent_held_equal_by_design"].astype(bool).any()
    assert reallocation["same_total_radius_reallocation"].astype(bool).all()


def test_tiny_celo_module_uses_real_gauge_autograd_jacobian_and_fresh_adam() -> None:
    vae = _TinyWeightVAE().float().eval()
    normalizer = WeightNormalizer(mean=torch.zeros(6), std=torch.ones(6))
    weights = torch.tensor([0.2, -0.5, 0.7, 0.1, -0.3, 0.4], dtype=torch.float32)
    z0, theta0 = _celo_literal_start(vae, normalizer, weights)
    jacobian = _celo_explicit_fp32_jacobian(vae, normalizer, z0, chunk_size=1)
    metric = jacobian.double().T @ jacobian.double()
    gauge = _whitening_gauge(metric, kind="full", rcond=1.0e-5)
    y0 = gauge.to_gauge(z0.double())

    grad_y, decoded, _loss = _gauge_loss_grad(
        y0,
        vae=vae,
        normalizer=normalizer,
        gauge=gauge,
        loss_fn=lambda theta: theta.square().sum() / 2.0,
    )
    z_leaf = z0.detach().clone().requires_grad_(True)
    direct = vae.decode_norm(z_leaf.reshape(1, -1)).squeeze(0)
    grad_z = torch.autograd.grad(direct.square().sum() / 2.0, z_leaf)[0]

    assert jacobian.dtype == torch.float32
    assert torch.equal(decoded, _celo_decode_gauged(vae, normalizer, y0, gauge))
    assert torch.allclose(decoded, theta0, atol=2e-6, rtol=2e-6)
    assert torch.allclose(grad_y, gauge.inverse.T @ grad_z.double(), atol=2e-6, rtol=2e-6)
    first_state = AdamState.zeros_like(y0)
    second_state = AdamState.zeros_like(y0)
    first_delta = _adam_delta(first_state, grad_y, lr=0.01)
    second_delta = _adam_delta(second_state, grad_y, lr=0.01)
    assert first_state.step == second_state.step == 1
    assert torch.equal(first_delta, second_delta)


def test_same_batch_validator_rejects_one_arm_hash_drift() -> None:
    diagnostics = pd.DataFrame(
        [
            {"vae_seed": "s0", "source_weight_index": 7, "proposal_step": 0, "arm_id": "a", "batch_indices_sha256": "batch"},
            {"vae_seed": "s0", "source_weight_index": 7, "proposal_step": 0, "arm_id": "b", "batch_indices_sha256": "batch"},
        ]
    )
    assert _validate_same_batches(diagnostics, expected_arm_ids=["a", "b"])["accepted"]
    diagnostics.loc[1, "batch_indices_sha256"] = "wrong"
    with pytest.raises(ValueError, match="same-batch protocol mismatch"):
        _validate_same_batches(diagnostics, expected_arm_ids=["a", "b"])


def test_trust_failure_is_retained_and_propagated_to_whole_trajectory() -> None:
    curves = pd.DataFrame(
        [
            {"vae_seed": "s0", "source_weight_index": 7, "arm_id": "a", "step": step}
            for step in range(3)
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "vae_seed": "s0",
                "source_weight_index": 7,
                "arm_id": "a",
                "proposal_step": step,
                "trust_executable": step == 0,
                "operator_executable": True,
                "retraction_executable": True,
                "treatment_executable": step == 0,
                "singularity_state_local": False,
            }
            for step in range(2)
        ]
    )
    propagated_curves, propagated_diagnostics, flags = _propagate_execution_flags(curves, diagnostics)

    assert len(propagated_diagnostics) == 2
    assert bool(flags.iloc[0]["trajectory_trust_failure"])
    assert not bool(flags.iloc[0]["trajectory_treatment_executable"])
    assert propagated_curves["trajectory_trust_failure"].astype(bool).all()


def test_coupled_normal_validator_requires_same_parent_tangent_scale_and_latent_delta() -> None:
    rows = []
    for arm_id in ("augmented_task_normal", "augmented_placebo"):
        rows.append(
            {
                "vae_seed": "s0",
                "source_weight_index": 7,
                "proposal_step": 0,
                "arm_id": arm_id,
                "state_z_sha256": "z",
                "paired_parent_theta_sha256": "parent",
                "tangent_proposal_sha256": "tangent",
                "executed_latent_delta_sha256": "dz",
                "trust_scale": 0.25,
                "full_realized_radius_error_task": 0.002,
                "full_realized_radius_error_placebo": 0.004,
                "pair_trust_executable": True,
                "pair_operator_executable": True,
                "pair_retraction_executable": True,
                "treatment_executable": True,
                "trust_bracket_found": True,
                "trust_scale_boundary_hit": False,
                "task_normal_norm": 2.0,
                "placebo_normal_norm": 2.0,
            }
        )
    diagnostics = pd.DataFrame(rows)
    assert _validate_coupled_normal_tangents(diagnostics)["accepted"]
    diagnostics.loc[diagnostics["arm_id"] == "augmented_placebo", "tangent_proposal_sha256"] = "drift"
    with pytest.raises(ValueError, match="does not hold tangent state fixed"):
        _validate_coupled_normal_tangents(diagnostics)


def test_old_tangent_only_scale_placebo_error_0p79_is_rejected() -> None:
    assert not _pair_trust_gate(
        task_error=0.002,
        placebo_error=0.79,
        bracket_found=True,
        boundary_hit=False,
    )
    rows = []
    for arm_id in ("augmented_task_normal", "augmented_placebo"):
        rows.append(
            {
                "vae_seed": "s0",
                "source_weight_index": 7,
                "proposal_step": 0,
                "arm_id": arm_id,
                "state_z_sha256": "z",
                "paired_parent_theta_sha256": "parent",
                "tangent_proposal_sha256": "tangent",
                "executed_latent_delta_sha256": "dz",
                "trust_scale": 0.25,
                "full_realized_radius_error_task": 0.002,
                "full_realized_radius_error_placebo": 0.79,
                "pair_trust_executable": True,
                "pair_operator_executable": True,
                "pair_retraction_executable": True,
                "treatment_executable": True,
                "trust_bracket_found": True,
                "trust_scale_boundary_hit": False,
                "task_normal_norm": 2.0,
                "placebo_normal_norm": 2.0,
            }
        )
    with pytest.raises(ValueError, match="full-radius trust validation failed"):
        _validate_coupled_normal_tangents(pd.DataFrame(rows))


def test_real_production_core_runs_tiny_metric_gauge_and_coupled_rollouts(tmp_path) -> None:
    context = _tiny_context(tmp_path)
    task, spec = _tiny_task_and_spec()
    task_tensors = {"tiny": task}
    generator = torch.Generator().manual_seed(29)
    weights = torch.randn(
        max((*PRODUCTION_FIT_SOURCES, *PRODUCTION_EVAL_SOURCES)) + 1,
        6,
        generator=generator,
    ) * 0.1
    fit_bank = pd.DataFrame(
        {
            "source_weight_index": PRODUCTION_FIT_SOURCES,
            "start_bank_position": range(16, 32),
            "task_name": ["tiny"] * 16,
            "tau": [1.0] * 16,
        }
    )
    eval_bank = pd.DataFrame({"source_weight_index": PRODUCTION_EVAL_SOURCES})
    metric, gauges, states, trajectory, arrays = _real_metric_fit(
        context=context,
        weights_cpu=weights,
        fit_bank=fit_bank,
        eval_bank=eval_bank,
        task_tensors=task_tensors,
        task_hashes=_task_hashes(task_tensors),
        spec=spec,
        device=torch.device("cpu"),
        batch_size=2,
        jacobian_chunk_size=1,
        progress_enabled=False,
    )

    assert metric.dtype == torch.float64
    assert len(states) == 16 * 4
    assert len(trajectory) == 16 * PRODUCTION_STEPS
    assert states["jacobian_dtype"].eq("torch.float32").all()
    assert states["gram_dtype"].eq("torch.float64").all()
    assert set(states["source_weight_index"]).isdisjoint(PRODUCTION_EVAL_SOURCES)
    assert "global_metric_fp64" in arrays

    source = int(PRODUCTION_EVAL_SOURCES[0])
    z0, theta0 = _celo_literal_start(context.vae, context.normalizer, weights[source])
    common = {
        "vae_seed": context.label,
        "source_weight_index": source,
        "start_bank_position": 0,
        "stream_start_index": 0,
        "task_name": "tiny",
        "tau": 1.0,
        "context_sha256": "context",
    }
    radii = [1.0e-3] * PRODUCTION_STEPS
    arm = Arm("z_adam:full:1e-5", "z_adam", "full", 1.0e-5, "pure_gauge")
    gauge_curves, gauge_diagnostics, occupancy = _real_gauge_curve(
        context=context,
        arm=arm,
        gauge=gauges[("full", "1e-5")],
        z0=z0,
        theta0=theta0,
        target_radii=radii,
        task_set=task,
        spec=spec,
        tau=1.0,
        stream_start_index=0,
        batch_size=2,
        jacobian_chunk_size=1,
        common=common,
    )
    local_curves, local_diagnostics = _real_coupled_local_curves(
        context=context,
        z0=z0,
        theta0=theta0,
        target_radii=radii,
        task_set=task,
        spec=spec,
        tau=1.0,
        stream_start_index=0,
        batch_size=2,
        jacobian_chunk_size=1,
        common=common,
        placebo_seed=31,
    )

    assert len(gauge_curves) == PRODUCTION_STEPS + 1
    assert len(gauge_diagnostics) == PRODUCTION_STEPS
    assert len(occupancy) == 4
    assert len(local_curves) == 3 * (PRODUCTION_STEPS + 1)
    assert len(local_diagnostics) == 3 * PRODUCTION_STEPS
    assert _validate_coupled_normal_tangents(pd.DataFrame(local_diagnostics))["accepted"]
    pair_diagnostics = pd.DataFrame(local_diagnostics)
    pair_diagnostics = pair_diagnostics[
        pair_diagnostics["arm_id"].isin(["augmented_task_normal", "augmented_placebo"])
    ]
    assert pair_diagnostics["proposal_policy"].eq("full_task_normal_radius_common_parent").all()
    assert pair_diagnostics["full_realized_radius_error_task"].max() <= 0.01
    expected_pair_trust = pair_diagnostics.apply(
        lambda row: _pair_trust_gate(
            task_error=float(row["full_realized_radius_error_task"]),
            placebo_error=float(row["full_realized_radius_error_placebo"]),
            bracket_found=bool(row["trust_bracket_found"]),
            boundary_hit=bool(row["trust_scale_boundary_hit"]),
        ),
        axis=1,
    )
    assert pair_diagnostics["pair_trust_executable"].astype(bool).eq(expected_pair_trust).all()
    combined = pd.concat(
        [pd.DataFrame(gauge_diagnostics), pd.DataFrame(local_diagnostics)],
        ignore_index=True,
        sort=False,
    )
    assert combined.groupby("proposal_step")["batch_indices_sha256"].nunique().max() == 1


def test_request_hash_rejects_elapsed_and_artifact_hash_detects_tampering(tmp_path) -> None:
    first = _request_hash({"protocol": "v2", "seed": 7})
    second = _request_hash({"seed": 7, "protocol": "v2"})
    assert first == second
    with pytest.raises(ValueError, match="exclude elapsed_sec"):
        _request_hash({"protocol": "v2", "elapsed_sec": 1.0})

    artifact = tmp_path / "artifact.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    expected = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
    assert _verify_artifact_hashes(tmp_path, expected)["accepted"]
    artifact.write_text("x\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        _verify_artifact_hashes(tmp_path, expected)


def test_real_smoke_preflight_selection_is_pinned_disjoint_and_nonproduction() -> None:
    selection = _validate_real_smoke_selection(
        run_values=["control"],
        labels=["control_seed0"],
        fit_source=PRODUCTION_FIT_SOURCES[0],
        eval_source=PRODUCTION_EVAL_SOURCES[0],
    )
    assert selection["accepted"]
    assert selection["production_grid"] is False
    assert selection["fit_source"] != selection["eval_source"]
    with pytest.raises(ValueError, match="rows16:31"):
        _validate_real_smoke_selection(
            run_values=["control"],
            labels=["control_seed0"],
            fit_source=PRODUCTION_EVAL_SOURCES[0],
            eval_source=PRODUCTION_EVAL_SOURCES[1],
        )
    limited_curves = pd.DataFrame(
        [
            {
                "vae_seed": "control_seed0",
                "source_weight_index": PRODUCTION_EVAL_SOURCES[0],
                "arm_id": "a",
                "step": step,
                "train_loss": 1.0 - 0.1 * step,
            }
            for step in range(REAL_SMOKE_STEPS + 1)
        ]
    )
    _per_start, _per_seed, aggregate = _primary_aggregation(
        limited_curves,
        steps=REAL_SMOKE_STEPS,
        expected_vae_seeds=("control_seed0",),
    )
    assert int(aggregate.iloc[0]["n_vae_seeds"]) == 1


def test_full_hash_index_writer_detects_post_write_tampering(tmp_path) -> None:
    first = tmp_path / "curves.csv"
    second = tmp_path / "validation.json"
    first.write_text("step,loss\n0,1.0\n", encoding="utf-8")
    second.write_text('{"accepted": true}\n', encoding="utf-8")
    hashes = {
        first.name: hashlib.sha256(first.read_bytes()).hexdigest(),
        second.name: hashlib.sha256(second.read_bytes()).hexdigest(),
    }
    index_path = _write_hash_index(tmp_path, file_hashes=hashes)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert bool(index["hash_index_self_excluded"])
    assert _verify_artifact_hashes(tmp_path, hashes)["accepted"]
    second.write_text('{"accepted": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        _verify_artifact_hashes(tmp_path, hashes)


def test_real_smoke_executor_wiring_writes_limited_nonproduction_bundle(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "tiny_control"
    run_dir.mkdir()
    for name in ("vae_checkpoint.pt", "config.json", "selected_lrs.csv"):
        (run_dir / name).write_bytes(name.encode("ascii"))
    context = _tiny_context(run_dir)
    task, spec = _tiny_task_and_spec()
    generator = torch.Generator().manual_seed(71)
    weights = torch.randn(
        max((*PRODUCTION_FIT_SOURCES, *PRODUCTION_EVAL_SOURCES)) + 1,
        6,
        generator=generator,
    ) * 0.1
    cfg = SimpleNamespace(
        device="cpu",
        dtype="float32",
        downstream_batch_size=2,
        show_progress=False,
        progress_backend="text",
    )
    checkpoint_payload = {"normalizer": context.normalizer.state_dict()}
    monkeypatch.setattr(gauge_harness, "_load_cfg", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        gauge_harness,
        "_load_weight_pool",
        lambda *_args, **_kwargs: (weights, pd.DataFrame([{"row": 1}]), "tiny_weight_key", spec),
    )
    monkeypatch.setattr(
        gauge_harness,
        "_load_task_tensors_for_pipeline",
        lambda *_args, **_kwargs: {"tiny": task},
    )
    monkeypatch.setattr(
        gauge_harness,
        "_load_vae",
        lambda *_args, **_kwargs: (context.vae, context.normalizer, checkpoint_payload),
    )
    monkeypatch.setattr(
        gauge_harness,
        "_selected_lr",
        lambda _run_dir, method: 1.0e-3 if method == "raw" else 1.0e-2,
    )
    fit_source = int(PRODUCTION_FIT_SOURCES[0])
    eval_source = int(PRODUCTION_EVAL_SOURCES[0])
    fit_bank = pd.DataFrame(
        [
            {
                "source_weight_index": fit_source,
                "start_bank_position": 16,
                "task_name": "tiny",
                "tau": 1.0,
                "protocol_split": "metric_fit",
            }
        ]
    )
    eval_bank = pd.DataFrame(
        [
            {
                "source_weight_index": eval_source,
                "start_bank_position": 0,
                "task_name": "tiny",
                "tau": 1.0,
                "protocol_split": "evaluation",
            }
        ]
    )
    output_dir = tmp_path / "real_smoke_output"
    validation = gauge_harness._run_real_smoke(
        output_dir=output_dir,
        run_value=str(run_dir),
        label="control_seed0",
        fit_bank=fit_bank,
        eval_bank=eval_bank,
        bank_hashes={"file_sha256": "bank", "evaluation_rows_sha256": "eval", "metric_fit_rows_sha256": "fit"},
        preflight={"request_hash": "preflight"},
        device_name="cpu",
        jacobian_chunk_size=1,
        seed=11,
        verbose=False,
        allow_cpu_for_tests=True,
    )

    assert validation["protocol_mode"] == "real_smoke"
    assert validation["real_smoke_acceptance_pass"]
    assert validation["production_acceptance_eligible"] is False
    assert validation["production_acceptance_pass"] is False
    curves = pd.read_csv(output_dir / "global_latent_gauge_curves.csv")
    diagnostics = pd.read_csv(output_dir / "global_latent_gauge_step_diagnostics.csv")
    assert len(curves) == 20 * (REAL_SMOKE_STEPS + 1)
    assert len(diagnostics) == 20 * REAL_SMOKE_STEPS
    runtime = json.loads((output_dir / "global_latent_gauge_real_smoke_runtime.json").read_text())
    assert runtime["counters"]["jacobian_calls"] > 0
    assert runtime["counters"]["decoder_forward_calls"] > 0
    index = json.loads((output_dir / "global_latent_gauge_output_hashes.json").read_text())
    assert _verify_artifact_hashes(output_dir, index["files"])["accepted"]
