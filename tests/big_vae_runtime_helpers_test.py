from __future__ import annotations

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from training.big_vae.runtime import _compute_model_latent_kl, _set_model_latent_sampling_gate
from training.big_vae.tracking import _build_external_tracking_params


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


def test_external_tracking_params_include_grad_clip_groups_without_name_error() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "grad_clip_norm_by_part": {
                    "distribution_encoder": 1.0,
                    "patch_tokenizer": 2.0,
                    "encoder": 3.0,
                    "decoder": 4.0,
                    "big_vae_other": 5.0,
                }
            },
            "model": {"big_vae": {"patch_tokenizer": {}}},
            "streaming": {},
            "collector": {},
        }
    )

    params = _build_external_tracking_params(cfg)

    assert params["train.grad_clip_norm_by_part.distribution_encoder"] == 1.0
    assert params["train.grad_clip_norm_by_part.patch_tokenizer"] == 2.0
    assert params["train.grad_clip_norm_by_part.encoder"] == 3.0
    assert params["train.grad_clip_norm_by_part.decoder"] == 4.0
    assert params["train.grad_clip_norm_by_part.big_vae_other"] == 5.0
