from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import analyze_kron_vae_operator_similarity as target


def _matrix_block(q_left: torch.Tensor, q_right: torch.Tensor) -> dict[str, object]:
    # The package expression for a rank-2 parameter: Q0^T Q0 @ G @ Q1^T Q1.
    return {
        "parameter_shape": (2, 2),
        "merged_shape": (2, 2),
        "q": [q_left, q_right],
        "exprs": ("unused", (), "an,bo,aA,bB,AB->no"),
    }


def test_factor_action_equals_dense_qtq_kron_action() -> None:
    # Use a direct package-compatible einsum expression rather than importing Kron at module load.
    q_left = torch.tensor([[1.0, 0.25], [0.0, 2.0]], dtype=torch.float64)
    q_right = torch.tensor([[1.5, 0.5], [0.0, 0.75]], dtype=torch.float64)
    block = _matrix_block(q_left, q_right)
    vector = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    got = target._block_action(block, vector).reshape(2, 2)
    expected = (q_left.T @ q_left) @ vector.reshape(2, 2) @ (q_right.T @ q_right)
    torch.testing.assert_close(got, expected)


def test_factor_action_equals_actual_kron_package_action_cpu() -> None:
    package = pytest.importorskip("kron_torch.kron")
    q_values, exprs = package._init_Q_exprs(
        torch.zeros(2, 2, dtype=torch.float32), 1.0, 8192, 2, None, dtype=torch.float32
    )
    q_values[0].copy_(torch.tensor([[1.0, 0.3], [0.0, 1.4]], dtype=torch.float32))
    q_values[1].copy_(torch.tensor([[0.9, -0.2], [0.0, 1.1]], dtype=torch.float32))
    block = {
        "parameter_shape": (2, 2),
        "merged_shape": (2, 2),
        "q": q_values,
        "exprs": exprs,
    }
    vector = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    actual = package._precond_grad(q_values, exprs, vector.reshape(2, 2)).reshape(-1)
    torch.testing.assert_close(target._block_action(block, vector), actual)


def test_trace_formulas_match_dense_toy_operators() -> None:
    q_left = torch.tensor([[1.0, 0.2], [0.0, 1.5]], dtype=torch.float64)
    q_right = torch.tensor([[0.8, -0.1], [0.0, 1.2]], dtype=torch.float64)
    block = _matrix_block(q_left, q_right)
    spectrum = target._operator_spectrum([block])
    # Build the exact dense operator from the action itself only for this 4-D unit test.
    basis = torch.eye(4, dtype=torch.float64)
    dense = torch.stack([target._operator_action([block], basis[:, idx]) for idx in range(4)], dim=1)
    jacobian = torch.tensor([[1.0, 0.0], [0.1, 1.0], [0.0, 0.3], [0.5, 0.2]], dtype=torch.float64)
    left, singular, _ = torch.linalg.svd(jacobian, full_matrices=False)
    kron_u = target._operator_action_matrix([block], left)
    summary, _ = target._operator_summary_for_rcond(
        kron_spectrum=spectrum,
        singular_values=singular,
        left_vectors=left,
        kron_u=kron_u,
        rcond=1.0e-6,
    )
    projector = left @ left.T
    jjt = jacobian @ jacobian.T
    assert summary["trace_kron_pullback"] == pytest.approx(float(torch.trace(dense @ projector)))
    assert summary["trace_kron_jjt"] == pytest.approx(float(torch.trace(dense @ jjt)))
    assert summary["frob_kron"] == pytest.approx(float(dense.norm()))
    assert np.sort(spectrum.detach().numpy()) == pytest.approx(np.sort(torch.linalg.eigvalsh(dense).numpy()))


def test_actual_update_metrics_do_not_apply_kron_a_second_time() -> None:
    # The realized descent is -delta/lr. It must be compared directly with
    # VAE(momentum), not with a second K(-delta/lr) transformation.
    momentum = torch.tensor([1.0, 1.0], dtype=torch.float64)
    actual_update = -target.SELECTED_LR * torch.tensor([2.0, 0.5], dtype=torch.float64)
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    jacobian = basis.clone()
    metrics = target._actual_update_metrics(
        actual_update=actual_update,
        momentum=momentum,
        basis=basis,
        jacobian=jacobian,
    )
    assert metrics["cos_kron_pullback"] == pytest.approx(2.0 / np.sqrt(2.0**2 + 0.5**2))
    assert metrics["kron_tangent_energy_fraction"] == pytest.approx(4.0 / 4.25)


def test_symmetry_error_uses_cauchy_scale_when_bilinear_form_cancels() -> None:
    # <a, b> is zero for these signed probes. Dividing by that bilinear form
    # would make an ordinary fp32 reduction residual spuriously huge.
    a = torch.tensor([1.0, 1.0], dtype=torch.float32)
    b = torch.tensor([1.0, -1.0], dtype=torch.float32)
    error = target._symmetry_relative_error(a, b, a, b)
    assert np.isfinite(error)
    assert error == pytest.approx(0.0)


def test_checkpoint_parser_rejects_terminal_and_keeps_sorted_unique() -> None:
    assert target._parse_checkpoint_steps(["75,0", "25", 25]) == (0, 25, 75)
    with pytest.raises(ValueError, match=r"\[0, 299\]"):
        target._parse_checkpoint_steps(["300"])
    with pytest.raises(ValueError, match="must not be empty"):
        target._parse_checkpoint_steps([])


def test_reference_row_comparison_handles_exact_and_numeric_failures() -> None:
    expected = {
        column: value
        for column, value in zip(target.EXACT_DIAGNOSTIC_COLUMNS, ["x"] * len(target.EXACT_DIAGNOSTIC_COLUMNS), strict=True)
    }
    expected.update({column: 1.0 for column in target.NUMERIC_DIAGNOSTIC_COLUMNS})
    actual = dict(expected)
    ok, failures, _ = target._compare_expected_row(actual, expected)
    assert ok and not failures
    actual["kron_q_sha256_after"] = "different"
    ok, failures, _ = target._compare_expected_row(actual, expected)
    assert not ok
    assert any("kron_q_sha256_after" in value for value in failures)


def test_spectrum_key_groups_do_not_mix_full_and_tangent_kron_spectra() -> None:
    arrays = {
        "sample_kron_eigenvalues": np.array([1.0, 2.0]),
        "sample_rcond_1e-5_tangent_kron_eigenvalues": np.array([1.0, 1.1]),
        "sample_rcond_1e-6_tangent_kron_eigenvalues": np.array([1.0, 1.2]),
        "sample_jacobian_singular_values": np.array([1.0, 0.1]),
    }
    full, tangent, jacobian = target._spectrum_key_groups(arrays)
    assert full == ["sample_kron_eigenvalues"]
    assert tangent == ["sample_rcond_1e-5_tangent_kron_eigenvalues"]
    assert jacobian == ["sample_jacobian_singular_values"]


def test_normalized_spectrum_band_rejects_nonpositive_and_normalizes_mean() -> None:
    quantiles, lower, median, upper = target._normalized_spectrum_band(
        [np.array([1.0, 3.0]), np.array([2.0, 6.0])]
    )
    assert len(quantiles) == len(lower) == len(median) == len(upper)
    assert median[0] == pytest.approx(0.5)
    assert median[-1] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="positive"):
        target._normalized_spectrum_band([np.array([0.0, 1.0])])


def test_selected_eval_rows_reject_source_leakage_and_nonprefix() -> None:
    eval_bank = pd.DataFrame(
        {
            "source_weight_index": [10, 11, 12],
            "global_stream_index": [8, 9, 10],
            "start_bank_position": [8, 9, 10],
            "split_start_index": [0, 1, 2],
            "protocol_split": ["eval", "eval", "eval"],
        }
    )
    tune_bank = pd.DataFrame({"source_weight_index": [1, 2]})
    selected = target._validate_selected_eval_rows(eval_bank, tune_bank, max_starts=2)
    assert selected["source_weight_index"].tolist() == [10, 11]
    with pytest.raises(ValueError, match="source leakage"):
        target._validate_selected_eval_rows(eval_bank, pd.DataFrame({"source_weight_index": [10]}), max_starts=2)
    broken = eval_bank.copy()
    broken.loc[1, "start_bank_position"] = 12
    with pytest.raises(ValueError, match="contiguous prefix"):
        target._validate_selected_eval_rows(broken, tune_bank, max_starts=2)
    broken_split = eval_bank.copy()
    broken_split.loc[1, "split_start_index"] = 4
    with pytest.raises(ValueError, match="split positions"):
        target._validate_selected_eval_rows(broken_split, tune_bank, max_starts=2)
