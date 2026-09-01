from __future__ import annotations

import torch
from torch import nn

from training.weightclip_benchmark.run_direct_normalized_scaled_700m_production import (
    STRICT_ENCODER_ONLY_SCOPE,
    _configure_parameter_training_scope,
)


class _TinyPolarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.continuous_projection = nn.Linear(2, 2, bias=False)
        self.scale_mlp = nn.Sequential(nn.Linear(1, 2), nn.SiLU(), nn.Linear(2, 2))
        self.group_embedding = nn.Embedding(2, 2)
        self.chunk_embedding = nn.Embedding(2, 2)
        self.tile_embedding = nn.Embedding(2, 2)
        self.distribution_encoder = nn.Linear(2, 2)
        self.encoder_blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.latent_slots = nn.Parameter(torch.zeros(1, 2))
        self.latent_norm = nn.LayerNorm(2)
        self.to_latent = nn.Linear(2, 1, bias=False)
        self.from_latent = nn.Linear(1, 2, bias=False)
        self.output_queries = nn.Parameter(torch.zeros(1, 2))
        self.decoder_query_conditioner = nn.Linear(2, 2)
        self.decoder_blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.direction_tail = nn.Linear(2, 2)
        self.scale_tail = nn.Linear(2, 2)
        self.direction_output_norm = nn.LayerNorm(2)
        self.scale_output_norm = nn.LayerNorm(2)
        self.direction_head = nn.Linear(2, 2, bias=False)
        self.scale_head = nn.Linear(2, 1, bias=False)


def test_strict_encoder_only_scope_freezes_all_direct_decoder_paths() -> None:
    model = _TinyPolarModel()
    ledger = _configure_parameter_training_scope(
        model,
        {"training_scope": STRICT_ENCODER_ONLY_SCOPE},
    )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert any(name.startswith("encoder_blocks.") for name in trainable)
    assert any(name.startswith("continuous_projection.") for name in trainable)
    assert any(name.startswith("to_latent.") for name in trainable)
    assert not any(name.startswith("decoder_blocks.") for name in trainable)
    assert not any(name.startswith("direction_tail.") for name in trainable)
    assert not any(name.startswith("scale_tail.") for name in trainable)
    assert not any(name.startswith("distribution_encoder.") for name in trainable)
    assert not any(name.startswith("tile_embedding.") for name in trainable)
    assert ledger["trainable_parameters"] + ledger["frozen_parameters"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
