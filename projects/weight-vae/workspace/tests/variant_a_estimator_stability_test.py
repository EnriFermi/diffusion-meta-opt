from __future__ import annotations

import numpy as np
import torch

from scripts.audit_variant_a_estimator_stability import (
    grid_bin,
    prefix_sum_scalar,
    prefix_sum_vector,
    symmetric_relative_error,
    vector_cosine,
    vector_relative_error,
)


def test_grid_bins_and_prefix_aggregation() -> None:
    endpoints = [1, 2, 4]
    assert [grid_bin(position, endpoints) for position in range(4)] == [0, 1, 2, 2]

    vector_bins = [
        [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]), torch.tensor([4.0, 0.0])],
        [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 2.0]), torch.tensor([0.0, 4.0])],
        [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]), torch.tensor([4.0, 4.0])],
    ]
    scalar_bins = np.asarray([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
    observed_vector = prefix_sum_vector(vector_bins, 1, 1, denominator=4.0)
    observed_scalar = prefix_sum_scalar(scalar_bins, 1, 1, denominator=4.0)
    assert torch.allclose(observed_vector, torch.tensor([0.75, 0.75]))
    assert observed_scalar == 1.5


def test_stability_metrics_handle_sign_and_scale() -> None:
    assert symmetric_relative_error(1.0, 1.0) == 0.0
    assert symmetric_relative_error(-1.0, 1.0) > 1.99
    assert symmetric_relative_error(0.0, 0.0) == 0.0

    reference = torch.tensor([1.0, 2.0, 3.0])
    assert abs(vector_cosine(reference, 2.0 * reference) - 1.0) < 1e-12
    assert abs(vector_relative_error(2.0 * reference, reference) - 1.0) < 1e-12
