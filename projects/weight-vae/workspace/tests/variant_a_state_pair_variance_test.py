from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.audit_variant_a_state_pair_variance import (
    Sufficient,
    _chunked_gram,
    _validate_shard_payload,
    variance_components_from_sufficient,
)


def test_variance_components_remove_pair_noise_from_state_means() -> None:
    # Two state effects (-1,+1) and two within-state residuals (-2,+2).
    gradients = torch.tensor([[-3.0], [1.0], [-1.0], [3.0]], dtype=torch.float32)
    state_means = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    within_residual_sum = float(((gradients[:2] - state_means[0]) ** 2).sum())
    within_residual_sum += float(((gradients[2:] - state_means[1]) ** 2).sum())
    sufficient = Sufficient(
        state_count=2,
        state_mean_sum=state_means.sum(dim=0),
        state_mean_squared_norm_sum=float((state_means**2).sum()),
        within_residual_sum=within_residual_sum,
        within_degrees_freedom=2,
    )

    result = variance_components_from_sufficient(sufficient, pair_count=2)

    assert result["within_pair_covariance_trace_W"] == 8.0
    assert result["state_mean_covariance_trace_V"] == 2.0
    assert result["between_state_covariance_trace_B_raw"] == -2.0
    assert result["within_pair_contribution_at_P_W_over_P"] == 4.0
    assert result["state_axis_dominates_at_P"] is False


def test_chunked_gram_matches_dense_float64_product() -> None:
    generator = torch.Generator().manual_seed(17)
    vectors = [torch.randn(19, generator=generator) for _ in range(4)]

    observed = _chunked_gram(vectors, chunk_size=5)
    dense = torch.stack(vectors).double()
    expected = (dense @ dense.T).numpy()

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def _valid_shard_payload() -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    identity: dict[str, object] = {"protocol": "test"}
    atomic_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    task_sums = {"mnist": 0.0, "fashion_mnist": 0.0}
    task_squared = {"mnist": 0.0, "fashion_mnist": 0.0}
    task_within = {"mnist": 0.0, "fashion_mnist": 0.0}
    for state in range(32):
        task = "mnist" if state % 2 == 0 else "fashion_mnist"
        state_mean = float(state + 1)
        task_sums[task] += state_mean
        task_squared[task] += state_mean * state_mean
        task_within[task] += 3.0
        atomic_squared_norm_sum = 4.0 * state_mean * state_mean + 3.0
        atomic_gradient_norm = float(np.sqrt(atomic_squared_norm_sum / 4.0))
        state_rows.append(
            {
                "repeat": 0,
                "state_position": state,
                "source_weight_index": 100 + state,
                "task_name": task,
                "tau": 1.0,
                "state_mean_a_p1": 1.0,
                "state_mean_a_p2": 1.0,
                "state_mean_a_p4": 1.0,
                "state_mean_gradient_norm_p4": state_mean,
                "state_mean_squared_gradient_norm_p4": state_mean * state_mean,
                "within_residual_sum": 3.0,
                "within_residual_sum_raw": 3.0,
                "normalized_within_residual_sum_raw": 3.0 / atomic_squared_norm_sum,
                "atomic_squared_gradient_norm_sum": atomic_squared_norm_sum,
            }
        )
        for pair in range(4):
            atomic_rows.append(
                {
                    "repeat": 0,
                    "state_position": state,
                    "pair_position": pair,
                    "source_weight_index": 100 + state,
                    "task_name": task,
                    "tau": 1.0,
                    "state_draw_seed": state + 1,
                    "train_position": state,
                    "probe_seed_1": 1000 + state * 8 + pair * 2,
                    "probe_seed_2": 1001 + state * 8 + pair * 2,
                    "a_scalar": 1.0,
                    "primary_a_scalar": 1.0,
                    "scalar_abs_error": 0.0,
                    "gradient_norm": atomic_gradient_norm,
                    "primary_gradient_norm": atomic_gradient_norm,
                    "gradient_norm_relative_error": 0.0,
                    "h1_norm": 1.0,
                    "h2_norm": 1.0,
                }
            )
    all_sum = sum(task_sums.values())
    all_squared = sum(task_squared.values())
    all_within = sum(task_within.values())
    groups = {
        "all": {
            "state_count": 32,
            "state_mean_sum": torch.tensor([all_sum]),
            "state_mean_squared_norm_sum": all_squared,
            "within_residual_sum": all_within,
            "within_degrees_freedom": 96,
        },
        **{
            task: {
                "state_count": 16,
                "state_mean_sum": torch.tensor([task_sums[task]]),
                "state_mean_squared_norm_sum": task_squared[task],
                "within_residual_sum": task_within[task],
                "within_degrees_freedom": 48,
            }
            for task in ("mnist", "fashion_mnist")
        },
    }
    payload: dict[str, object] = {
        "shard_identity": identity,
        "repeat": 0,
        "atomic_rows": atomic_rows,
        "state_rows": state_rows,
        "prefix_gradient_sums": {
            1: torch.tensor([1.0]),
            2: torch.tensor([2.0]),
            4: torch.tensor([4.0 * all_sum]),
        },
        "prefix_scalar_sums": {1: 32.0, 2: 64.0, 4: 128.0},
        "group_sufficient": groups,
        "replay_summary": {
            "max_scalar_abs_error": 0.0,
            "max_gradient_norm_relative_error": 0.0,
            "minimum_raw_within_residual": 3.0,
            "minimum_normalized_within_residual": min(
                float(row["normalized_within_residual_sum_raw"]) for row in state_rows
            ),
        },
    }
    primary = pd.DataFrame(atomic_rows).set_index(["repeat", "state_position", "pair_position"])
    return payload, identity, primary


def test_resume_payload_rejects_nonfinite_rows() -> None:
    payload, identity, primary = _valid_shard_payload()
    _validate_shard_payload(
        payload, shard_identity=identity, primary_atomic=primary, repeat=0, dimension=1
    )
    payload["atomic_rows"][0]["scalar_abs_error"] = float("nan")  # type: ignore[index]

    with pytest.raises(FloatingPointError):
        _validate_shard_payload(
            payload, shard_identity=identity, primary_atomic=primary, repeat=0, dimension=1
        )


def test_resume_payload_rejects_corrupt_sufficient_statistic() -> None:
    payload, identity, primary = _valid_shard_payload()
    payload["group_sufficient"]["all"]["within_residual_sum"] = 1e12  # type: ignore[index]

    with pytest.raises(ValueError, match="within sum mismatch"):
        _validate_shard_payload(
            payload, shard_identity=identity, primary_atomic=primary, repeat=0, dimension=1
        )


def test_resume_payload_rejects_primary_replay_drift() -> None:
    payload, identity, primary = _valid_shard_payload()
    payload["atomic_rows"][0]["a_scalar"] = 9.0  # type: ignore[index]

    with pytest.raises(ValueError, match="primary replay values mismatch"):
        _validate_shard_payload(
            payload, shard_identity=identity, primary_atomic=primary, repeat=0, dimension=1
        )


def test_resume_payload_rejects_primary_source_drift() -> None:
    payload, identity, primary = _valid_shard_payload()
    payload["atomic_rows"][0]["source_weight_index"] = 999_999  # type: ignore[index]

    with pytest.raises(ValueError, match="source_weight_index mismatch"):
        _validate_shard_payload(
            payload, shard_identity=identity, primary_atomic=primary, repeat=0, dimension=1
        )


def test_resume_payload_rejects_within_identity_drift() -> None:
    payload, identity, primary = _valid_shard_payload()
    payload["state_rows"][0]["within_residual_sum"] = 9.0  # type: ignore[index]
    payload["group_sufficient"]["all"]["within_residual_sum"] += 6.0  # type: ignore[index,operator]
    payload["group_sufficient"]["mnist"]["within_residual_sum"] += 6.0  # type: ignore[index,operator]

    with pytest.raises(ValueError, match="within-state identity mismatch"):
        _validate_shard_payload(
            payload, shard_identity=identity, primary_atomic=primary, repeat=0, dimension=1
        )
