from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.big_vae import presliced as presliced_runtime


def test_next_valid_presliced_slice_reads_valid_record() -> None:
    sample = SimpleNamespace(
        x=torch.ones((3, 4), dtype=torch.float32),
        weight=torch.ones((4, 2), dtype=torch.float32),
        meta={},
        model_name="model",
        layer_name="layer",
    )

    record = presliced_runtime._next_valid_presliced_slice(
        iter([sample]),
        logging.getLogger("test_next_valid_presliced_slice_reads_valid_record"),
    )

    assert tuple(record.x.shape) == (3, 4)
    assert tuple(record.W.shape) == (4, 2)
    assert tuple(record.x_mask.shape) == (3,)
    assert tuple(record.d_in_mask.shape) == (4,)
    assert tuple(record.d_out_mask.shape) == (2,)
    assert record.model_name == "model"
    assert record.layer_name == "layer"


def test_next_valid_presliced_slice_does_not_swallow_programmer_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = SimpleNamespace(
        x=torch.ones((3, 4), dtype=torch.float32),
        weight=torch.ones((4, 2), dtype=torch.float32),
        meta={},
    )

    def raise_name_error(tensor: torch.Tensor) -> torch.Tensor:
        del tensor
        raise NameError("missing helper")

    monkeypatch.setattr(presliced_runtime, "_prepare_cpu_sample_tensor", raise_name_error)

    with pytest.raises(NameError, match="missing helper"):
        presliced_runtime._next_valid_presliced_slice(
            iter([sample]),
            logging.getLogger("test_next_valid_presliced_slice_does_not_swallow_programmer_errors"),
        )
