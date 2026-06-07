from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.big_vae.runtime import _compute_model_latent_kl, _set_model_latent_sampling_gate


class _DummyLatentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate: float | None = None

    def set_latent_sampling_gate(self, gate: float) -> None:
        self.gate = float(gate)

    def latent_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        del mu, logvar
        return torch.tensor(7.0)


def test_runtime_latent_helpers_use_runtime_model_unwrap() -> None:
    model = _DummyLatentModel()

    _set_model_latent_sampling_gate(model, 0.25)
    kl = _compute_model_latent_kl(model, torch.zeros(1, 2), torch.zeros(1, 2))

    assert model.gate == 0.25
    assert float(kl) == 7.0
