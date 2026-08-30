from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import torch

import scripts.run_one_state_a_full_burg_p32_training as p32


def test_p32_builder_pools_before_one_clip_and_one_adam(monkeypatch) -> None:
    active = [
        torch.nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([1.2], dtype=torch.float32)),
    ]

    def fake_pair_losses(
        *,
        probe_generator_1: torch.Generator,
        probe_generator_2: torch.Generator,
        **_kwargs,
    ):
        coefficients_a = torch.randn(3, generator=probe_generator_1)
        coefficients_b = torch.randn(3, generator=probe_generator_2)
        flat = torch.cat([parameter.reshape(-1) for parameter in active])
        a_loss = (flat * coefficients_a).sum()
        b_loss = (flat * coefficients_b).sum()
        return a_loss, b_loss, {}

    real_clip = torch.nn.utils.clip_grad_norm_
    clip_calls = 0

    def counted_clip(*args, **kwargs):
        nonlocal clip_calls
        clip_calls += 1
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(p32, "_pair_losses", fake_pair_losses)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    p32.DRAW_ROWS.clear()
    p32.PAIR_ROWS.clear()
    p32.POOLING_PREFLIGHT.clear()
    p32.TREATMENT_BUILD_CALLS = 0
    p32.TREATMENT_CLIP_CALLS = 0
    p32.TREATMENT_ADAM_CALLS = 0

    result = p32._build_p32_proposal(
        cfg=SimpleNamespace(vae_precond_hvp_mode="autograd"),
        run=None,
        z=torch.zeros(1),
        record={},
        active=active,
        burg_gradient=torch.zeros(1),
        beta=1.7,
        proposal=1,
        pairs=4,
        exp_avg=[torch.zeros_like(parameter) for parameter in active],
        exp_avg_sq=[torch.zeros_like(parameter) for parameter in active],
        accepted_adam_step=0,
        gradient_clip=1.0,
        lr=3e-5,
    )

    assert len(p32.DRAW_ROWS) == 8
    assert len(p32.PAIR_ROWS) == 32
    assert len({(row["draw"], row["pair"]) for row in p32.PAIR_ROWS}) == 32
    seeds = [
        int(row[key])
        for row in p32.PAIR_ROWS
        for key in ("seed_1", "seed_2")
    ]
    assert len(set(seeds)) == 64
    assert max(
        float(value)
        for key, value in p32.POOLING_PREFLIGHT.items()
        if key.endswith("relative_error")
    ) <= 1e-6
    assert p32.TREATMENT_BUILD_CALLS == 1
    assert p32.TREATMENT_CLIP_CALLS == 1
    assert p32.TREATMENT_ADAM_CALLS == 1
    assert clip_calls == 1
    assert result["proposal_norm"] > 0.0
    assert result["preclip_norm"] > 0.0
    assert all(parameter.grad is None for parameter in active)


def test_seed_schedule_survives_mixed_csv_roundtrip(tmp_path: Path) -> None:
    expected = []
    rows = []
    for proposal in range(1, 101):
        for draw in range(p32.POOL_DRAWS):
            for pair in range(p32.PAIRS_PER_DRAW):
                row = {
                    "proposal": proposal,
                    "draw": draw,
                    "pair": pair,
                    "seed_1": p32._branch_seed(proposal, draw, pair, 0),
                    "seed_2": p32._branch_seed(proposal, draw, pair, 1),
                }
                expected.append(row)
                rows.append({**row, "a_loss": float(proposal) / 7.0})

    path = tmp_path / "mixed_seed_rows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    observed = p32._schedule_from_pair_frame(pd.read_csv(path))
    preflight = p32._seed_schedule_preflight()

    assert observed == expected
    assert p32._seed_schedule_hash(observed) == preflight["schedule_sha256"]
