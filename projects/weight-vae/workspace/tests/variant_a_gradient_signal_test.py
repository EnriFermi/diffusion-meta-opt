from __future__ import annotations

import numpy as np

from scripts.analyze_variant_a_gradient_signal import cross_repeat_signal_estimate, jackknife_signal


def test_cross_repeat_signal_recovers_shared_scalar_mean() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    gram = np.outer(values, values)
    expected = sum(values[left] * values[right] for left in range(4) for right in range(4) if left != right) / 12.0

    observed = cross_repeat_signal_estimate(gram)
    estimate, standard_error, lower, upper = jackknife_signal(gram)

    assert observed == expected
    assert estimate == expected
    assert standard_error > 0.0
    assert lower < estimate < upper
