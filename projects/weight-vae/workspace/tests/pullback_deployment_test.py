from __future__ import annotations

import pandas as pd
import pytest
import torch

from scripts.audit_latent_raw_common_state_rate import _build_projector, _validate_adam
from scripts.evaluate_pullback_deployment_intervention import (
    _branch_key_set,
    _context_mismatch_count,
    _executed_preimage_residuals,
    _expected_eval_steps,
    _first_upward_crossings,
    _gap_closure_summary,
    _is_retraction_executable,
    _paired_contrasts,
    _propagate_rollout_singularity,
    _select_protocol_start_rows,
)


def test_first_upward_crossing_precedes_later_crossing() -> None:
    observations = [(0.0, 0.0), (0.25, 0.4), (0.5, 0.1), (1.0, 0.6)]

    crossings = _first_upward_crossings(observations, 0.3)

    assert len(crossings) == 2
    assert crossings[0] == ((0.0, 0.0), (0.25, 0.4))


def test_executed_preimage_residual_checks_realized_latent_increment() -> None:
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]], dtype=torch.float32)
    projector = _build_projector(jacobian, rcond=1.0e-5)
    ambient = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    scale = 0.125
    ideal_executed = (projector.latent_preimage64(ambient) * scale).float()

    valid = _executed_preimage_residuals(projector, ambient, ideal_executed, scale=scale)
    lost_to_state_addition = _executed_preimage_residuals(
        projector,
        ambient,
        torch.zeros_like(ideal_executed),
        scale=scale,
    )

    assert valid["executable"] <= 1.0e-6
    assert lost_to_state_addition["executable"] > 0.99


def test_exact_branch_key_set_detects_replaced_branch() -> None:
    expected = {
        ("seed0", 7, "raw_adam", 1.0, 0),
        ("seed0", 7, "pullback_natural_trust", 0.25, 0),
    }
    frame = pd.DataFrame(
        {
            "label": ["seed0", "seed0"],
            "source_weight_index": [7, 7],
            "method": ["raw_adam", "pullback_natural_trust"],
            "trust_fraction": [1.0, 0.25],
            "step": [0, 0],
        }
    )
    assert _branch_key_set(frame, step_column="step") == expected

    replaced = frame.copy()
    replaced.loc[1, "method"] = "latent_sgd_jjt_trust"
    actual = _branch_key_set(replaced, step_column="step")
    assert len(expected - actual) == 1
    assert len(actual - expected) == 1


def test_start_context_rejects_substituted_source() -> None:
    bank = pd.DataFrame(
        {
            "source_weight_index": [7],
            "start_bank_position": [0],
            "task_name": ["mnist"],
            "tau": [1.0],
        }
    )
    frame = pd.DataFrame(
        {
            "label": ["seed0"],
            "source_weight_index": [7],
            "start_bank_position": [0],
            "stream_start_index": [0],
            "task_name": ["mnist"],
            "tau": [1.0],
        }
    )
    assert _context_mismatch_count(frame, bank, ["seed0"]) == 0

    frame.loc[0, "source_weight_index"] = 8
    assert _context_mismatch_count(frame, bank, ["seed0"]) == 1


def test_eval_steps_include_terminal_remainder() -> None:
    assert _expected_eval_steps(5, 2) == [0, 2, 4, 5]


def test_manual_adam_native_parity_cpu() -> None:
    assert _validate_adam(torch.device("cpu")) <= 2.0e-7


def test_retraction_gate_applies_only_to_local_linearized_arms() -> None:
    assert _is_retraction_executable("pullback_natural_trust", 0.25, 0.95)
    assert not _is_retraction_executable("pullback_natural_trust", 0.251, 0.99)
    assert not _is_retraction_executable("latent_sgd_jjt_trust", 0.1, 0.949)
    assert _is_retraction_executable("latent_adam_trust", float("nan"), float("nan"))


def test_rollout_singularity_propagates_to_every_method_and_step() -> None:
    diagnostics = pd.DataFrame(
        {
            "label": ["seed0", "seed0", "seed0"],
            "source_weight_index": [6026, 6026, 7],
            "method": ["raw_adam", "pullback_natural_trust", "raw_adam"],
            "singularity_state_local": [False, True, False],
        }
    )

    propagated = _propagate_rollout_singularity(diagnostics)

    assert propagated.loc[propagated["source_weight_index"] == 6026, "singularity_stratum"].all()
    assert not propagated.loc[propagated["source_weight_index"] == 7, "singularity_stratum"].any()


def test_production_selects_predeclared_first_16_from_hash_pinned_64_bank() -> None:
    bank = pd.DataFrame(
        {
            "source_weight_index": list(range(64)),
            "start_bank_position": list(range(64)),
        }
    )

    selected = _select_protocol_start_rows(bank, protocol_mode="production", max_starts=0)

    assert selected["source_weight_index"].tolist() == list(range(16))
    assert selected["start_bank_position"].tolist() == list(range(16))


def test_paired_contrasts_average_starts_before_seed_inference(tmp_path) -> None:
    branches = [
        ("raw_adam", 1.0),
        ("latent_adam_native", 1.0),
        ("latent_adam_trust", 0.25),
        ("latent_adam_trust", 1.0),
        ("latent_sgd_jjt_trust", 0.25),
        ("latent_sgd_jjt_trust", 1.0),
        ("pullback_natural_trust", 0.25),
        ("pullback_natural_trust", 1.0),
        ("projected_raw_adam_trust", 0.25),
        ("projected_raw_adam_trust", 1.0),
    ]
    rows = []
    for seed in range(3):
        for source in (7, 8):
            for method, fraction in branches:
                value = float(seed + source) + (1.0 if method == "pullback_natural_trust" else 0.0)
                for metric in ("train_loss", "test_loss"):
                    rows.append(
                        {
                            "label": f"seed{seed}",
                            "source_weight_index": source,
                            "method": method,
                            "trust_fraction": fraction,
                            "metric": metric,
                            "post0_progress": value,
                            "final_progress": value,
                            "ols_slope": -value,
                            "trajectory_treatment_executable": True,
                            "trajectory_singularity_stratum": False,
                        }
                    )

    summary = _paired_contrasts(pd.DataFrame(rows), tmp_path, [0.25, 1.0])
    clustered = summary[
        (summary["contrast"] == "pullback_f0.25_minus_jjt")
        & (summary["metric"] == "train_loss")
        & (summary["score"] == "post0_progress")
        & (summary["analysis_set"] == "all_start_deployment_itt")
        & (summary["inference_scope"] == "vae_seed_clustered")
    ]

    assert len(clustered) == 1
    assert int(clustered.iloc[0]["n_vae_seeds"]) == 3
    assert float(clustered.iloc[0]["mean_left_better_positive"]) == 1.0


def test_gap_closure_uses_predeclared_fraction_and_80pct_threshold(tmp_path) -> None:
    progress = {
        "raw_adam": 2.0,
        "latent_adam_trust": 0.0,
        "latent_sgd_jjt_trust": 0.5,
        "pullback_natural_trust": 1.6,
    }
    rows = []
    for seed in range(3):
        for method, value in progress.items():
            rows.append(
                {
                    "label": f"seed{seed}",
                    "source_weight_index": 7,
                    "method": method,
                    "trust_fraction": 1.0,
                    "metric": "train_loss",
                    "post0_progress": value,
                }
            )

    summary = _gap_closure_summary(pd.DataFrame(rows), tmp_path, production_mode=True)
    aggregate = summary[summary["scope"] == "across_vae_seed_means"].iloc[0]

    assert float(aggregate["gap_recovery_fraction"]) == pytest.approx(0.8)
    assert bool(aggregate["predeclared_rc5c_sufficiency_success"])
