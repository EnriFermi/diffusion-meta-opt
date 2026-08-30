from __future__ import annotations

import numpy as np

from scripts.bootstrap_variant_a_scalar_stability import bootstrap, evaluate_repeat_means


def test_constant_repeat_means_pass_all_scalar_gates() -> None:
    repeat_means = np.full((5, 12), 3.0, dtype=np.float64)

    reference_pass, cell_pass, max_split_error, cell_q90 = evaluate_repeat_means(repeat_means)

    assert reference_pass.all()
    assert cell_pass.all()
    assert np.array_equal(max_split_error, np.zeros(5))
    assert np.array_equal(cell_q90, np.zeros(5))


def test_constant_cluster_population_has_unit_pass_probability() -> None:
    result = bootstrap(
        np.ones(8, dtype=np.float64),
        state_counts=(2, 8),
        trials=50,
        seed=7,
        chunk_size=10,
    )

    assert (result["reference_gate_pass_probability"] == 1.0).all()
    assert (result["cell_scalar_gate_pass_probability"] == 1.0).all()
    assert (result["both_scalar_gates_pass_probability"] == 1.0).all()
