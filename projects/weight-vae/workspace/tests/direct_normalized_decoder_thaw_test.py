from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    POLAR_TAIL_ENCODER_WARM_DECODER_RAMP_SCHEMA,
    STRICT_ENCODER_ONLY_SCOPE,
    _build_production_optimizer,
    _configure_parameter_training_scope,
    _decoder_thaw_groups_by_name,
    _load_config,
    _restore_encoder_only_optimizer_state_for_decoder_thaw,
    _set_decoder_thaw_learning_rates,
)
from tests.direct_normalized_encoder_only_scope_test import _TinyPolarModel


CONFIG_PATH = Path(__file__).resolve().parents[1] / Path(
    "conf/weightclip_benchmark/"
    "direct_normalized_scaled_700m_polar_tails_"
    "encoder_warm_decoder_ramp10k_production_500k.yaml"
)


def _tiny_thaw_config() -> dict[str, object]:
    return {
        "schema": POLAR_TAIL_ENCODER_WARM_DECODER_RAMP_SCHEMA,
        "learning_rate": 5.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "decoder_thaw": {
            "encoder_learning_rate": 5.0e-5,
            "decoder_start_learning_rate": 0.0,
            "decoder_target_learning_rate": 5.0e-5,
            "ramp_steps": 10_000,
        },
    }


def test_decoder_thaw_schedule_hits_declared_endpoints() -> None:
    model = _TinyPolarModel()
    optimizer = _build_production_optimizer(model, _tiny_thaw_config())

    expected = {
        0: 0.0,
        1: 5.0e-9,
        5_000: 2.5e-5,
        10_000: 5.0e-5,
        20_000: 5.0e-5,
    }
    for step, expected_decoder_lr in expected.items():
        state = _set_decoder_thaw_learning_rates(
            optimizer,
            _tiny_thaw_config(),
            step=step,
        )
        groups = _decoder_thaw_groups_by_name(optimizer)
        assert groups["encoder_continuation"]["lr"] == 5.0e-5
        assert groups["decoder_linear_thaw"]["lr"] == pytest.approx(
            expected_decoder_lr
        )
        assert state["decoder_learning_rate"] == pytest.approx(expected_decoder_lr)


def test_encoder_adam_is_preserved_and_decoder_adam_starts_empty() -> None:
    source_model = _TinyPolarModel()
    _configure_parameter_training_scope(
        source_model,
        {"training_scope": STRICT_ENCODER_ONLY_SCOPE},
    )
    source_parameters = [
        parameter for parameter in source_model.parameters() if parameter.requires_grad
    ]
    source_optimizer = torch.optim.AdamW(source_parameters, lr=5.0e-5)
    for parameter in source_parameters:
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    source_state = source_optimizer.state_dict()

    target_model = _TinyPolarModel()
    target_optimizer = _build_production_optimizer(
        target_model,
        _tiny_thaw_config(),
    )
    stats = _restore_encoder_only_optimizer_state_for_decoder_thaw(
        target_optimizer,
        source_state,
    )
    groups = _decoder_thaw_groups_by_name(target_optimizer)
    encoder_parameters = groups["encoder_continuation"]["params"]
    decoder_parameters = groups["decoder_linear_thaw"]["params"]

    assert stats["encoder_optimizer_state_entries"] == len(source_parameters)
    assert stats["decoder_optimizer_state_entries"] == 0
    assert all(parameter in target_optimizer.state for parameter in encoder_parameters)
    assert all(parameter not in target_optimizer.state for parameter in decoder_parameters)
    for source_parameter, target_parameter in zip(
        source_parameters,
        encoder_parameters,
        strict=True,
    ):
        assert torch.equal(
            source_optimizer.state[source_parameter]["exp_avg"],
            target_optimizer.state[target_parameter]["exp_avg"],
        )


def test_production_decoder_thaw_config_contract() -> None:
    config = _load_config(CONFIG_PATH)
    assert config["device"] == "cuda:0"
    assert config["decoder_thaw"]["source_step"] == 13_000
    assert config["decoder_thaw"]["source_committed_logical_index"] == 416_000
    assert config["decoder_thaw"]["ramp_steps"] == 10_000
