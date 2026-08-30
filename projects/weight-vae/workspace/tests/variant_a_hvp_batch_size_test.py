from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scripts.audit_variant_a_hvp_batch_size import (
    BatchResult,
    analyze,
    circular_mask,
    gram_group_metrics,
    gram_pair_metrics,
    window_diagnostics,
)
from scripts.review_variant_a_hvp_batch_size import concatenated_gram_metrics, expected_batch


def test_circular_mask_wraps_and_full_mode() -> None:
    observed = circular_mask(6, batch_size=4, train_count=8)
    assert observed.tolist() == [True, True, False, False, False, False, True, True]
    assert circular_mask(None, batch_size=8, train_count=8).all()


def test_independent_batch_replay_matches_known_offset() -> None:
    offset, digest = expected_batch(batch_size=128, step=10, sample_key=10954, pair_key=0)
    assert offset == (10954 * 1009 + 10 * 9176) % 16384
    assert isinstance(digest, str) and len(digest) == 64
    assert expected_batch(batch_size=16384, step=10, sample_key=10954, pair_key=0) == (None, "full")


def test_window_diagnostics_reports_union_and_overlap() -> None:
    branches = []
    for pair in range(4):
        for branch in range(2):
            branches.append(
                {
                    "pair_position": pair,
                    "branch": branch,
                    "offset": (2 * pair + branch) % 8,
                    "batch_hash": f"{pair}-{branch}",
                    "train_count": 16384,
                    "effective_batch_size": 2,
                    "batch_mode": "window",
                }
            )
    coverage, overlaps = window_diagnostics(
        branches,
        requested_batch_size=2,
        repeat=0,
        state_position=0,
        source_weight_index=1,
        task_name="mnist",
        step_key=10,
    )
    assert coverage["branch_count"] == 8
    assert len(overlaps) == 28
    assert coverage["union_count"] == 9
    assert coverage["max_multiplicity"] == 2


def test_gram_pair_and_group_metrics() -> None:
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    gram = vectors @ vectors.T
    cosine, relative_error, norm_ratio = gram_pair_metrics(gram, 1, 0)
    assert abs(cosine - 1.0) < 1e-12
    assert abs(relative_error - 1.0) < 1e-12
    assert abs(norm_ratio - 2.0) < 1e-12

    cosine, relative_error, norm_ratio = gram_group_metrics(gram, [0, 1], [2, 3])
    assert abs(cosine) < 1e-12
    assert abs(relative_error - np.sqrt(2.0)) < 1e-12
    assert abs(norm_ratio - 1.0) < 1e-12


def test_concatenated_gram_metrics_weights_by_gradient_energy() -> None:
    left = np.asarray([[0.8, 0.0], [0.0, 10.0]])
    right = np.asarray([[1.0, 0.0], [0.0, 10.0]])
    vectors = np.concatenate([left, right], axis=0)
    gram = vectors @ vectors.T
    cosine, relative_error, norm_ratio = concatenated_gram_metrics(gram, [0, 1], [2, 3])
    expected_relative_error = 0.2 / np.sqrt(101.0)
    assert cosine > 0.999
    assert abs(relative_error - expected_relative_error) < 1e-12
    assert norm_ratio < 1.0


def test_analyze_small_paired_fixture() -> None:
    results = {}
    all_vectors = []
    gradient_rows = []
    for batch_size, scale in ((128, 0.8), (16384, 1.0)):
        gradients = [torch.tensor([scale, 0.0]), torch.tensor([0.0, scale])]
        scalars = [1.0 + scale, 2.0 + scale]
        atomic_rows = []
        coverage_rows = []
        for repeat in range(2):
            for state in range(2):
                coverage_rows.append(
                    {
                        "requested_batch_size": batch_size,
                        "union_fraction": 1.0 if batch_size == 16384 else 0.5,
                        "max_multiplicity": 8 if batch_size == 16384 else 2,
                        "fraction_multiplicity_ge2": 1.0 if batch_size == 16384 else 0.1,
                    }
                )
                for pair in range(4):
                    atomic_rows.append(
                        {
                            "requested_batch_size": batch_size,
                            "repeat": repeat,
                            "state_position": state,
                            "pair_position": pair,
                            "task_name": "mnist",
                            "a_scalar": scalars[repeat] + 0.01 * (4 * state + pair),
                            "gradient_norm": scale + 0.01 * pair,
                            "unit_elapsed_sec": 0.1,
                            "cuda_peak_allocated_bytes": 100,
                            "cuda_peak_reserved_bytes": 200,
                        }
                    )
            gradient_rows.append(
                {
                    "gradient_index": len(all_vectors),
                    "requested_batch_size": batch_size,
                    "repeat": repeat,
                }
            )
            all_vectors.append(gradients[repeat].numpy())
        results[batch_size] = BatchResult(
            requested_batch_size=batch_size,
            effective_batch_size=batch_size,
            batch_mode="full" if batch_size == 16384 else "window",
            gradients=gradients,
            scalars=scalars,
            atomic_rows=atomic_rows,
            coverage_rows=coverage_rows,
            overlap_rows=[],
            elapsed_sec=1.0,
        )
    matrix = np.asarray(all_vectors)
    gram = matrix @ matrix.T
    bundle = analyze(
        results,
        full_gram=gram,
        module_grams={"all": gram},
        gradient_index=pd.DataFrame(gradient_rows),
        reference_batch_size=16384,
        seed=1,
    )
    summary = bundle["summary"].set_index("requested_batch_size")
    assert abs(summary.loc[128, "paired_gradient_cosine_median"] - 1.0) < 1e-12
    assert abs(summary.loc[128, "paired_gradient_norm_ratio_median"] - 0.8) < 1e-6
    assert len(bundle["task_summary"]) == 2
