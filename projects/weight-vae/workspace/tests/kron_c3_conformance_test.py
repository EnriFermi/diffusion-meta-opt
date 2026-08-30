from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts import audit_kron_c3_conformance as audit


def _fixture() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    q = [
        torch.tensor([[1.2, 0.25], [0.0, 0.8]], dtype=dtype),
        torch.tensor([[0.9, -0.1], [0.0, 1.3]], dtype=dtype),
    ]
    g = torch.tensor([[0.7, -0.2], [0.3, 0.5]], dtype=dtype)
    v = torch.tensor([[0.4, -0.8], [1.1, 0.2]], dtype=dtype)
    return q, g, v


def test_fixed_V_A_B_S_and_lie_reference_conform() -> None:
    q, g, v = _fixture()
    package = audit._package_pair(q, g, v)
    reference = audit._reference_pair(q, g, v)
    audit._assert_term_conformance(package, reference)
    assert audit._pair_and_lie_check()["max_delta_error"] < 2e-10


def test_negative_transpose_convention_control_fails() -> None:
    q, g, v = _fixture()
    package = audit._package_pair(q, g, v)
    perturbed = g + audit._rho(g) * v
    wrong = (q[0] @ perturbed.T @ q[1].T, audit._reference_pair(q, g, v)[1])
    with pytest.raises(AssertionError):
        audit._assert_pair_terms(package, wrong)


def test_negative_redrawn_V_control_fails() -> None:
    q, g, v = _fixture()
    package = audit._package_pair(q, g, v)
    redrawn_v = torch.tensor([[-0.9, 0.1], [0.2, -1.3]], dtype=torch.float64)
    with pytest.raises(AssertionError):
        audit._assert_pair_terms(package, audit._reference_pair(q, g, redrawn_v))


def test_compile_is_disabled_and_negative_controls_are_real() -> None:
    audit._load_package()
    assert torch.compile is audit._identity_compile
    with pytest.raises(AssertionError):
        audit._assert_term_conformance(
            audit._package_pair(*_fixture()),
            audit._reference_pair(_fixture()[0], _fixture()[1], _fixture()[2].T),
        )


def test_full_cpu_audit_writes_validation_and_details(tmp_path: Path) -> None:
    result = audit.run_audit(tmp_path, seed=17, quiet=True)
    assert result["accepted"] is True
    assert result["scope"] == "practical package criterion-conformance; not exact Li c3"
    assert Path(result["artifacts"]["json"]).is_file()
    assert Path(result["artifacts"]["csv"]).is_file()
    assert result["checks"]["synthetic_3d_merge_restore"]["passed"] is True
