import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / ".LSDL" / "vq_codebook_utilization_fsq_lfq.ipynb"


def load_lfq_class():
    notebook = json.loads(NOTEBOOK.read_text())
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "class LFQ(nn.Module)" in "".join(cell.get("source", []))
    )
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(source, f"{NOTEBOOK}:LFQ", "exec"), namespace)
    return namespace["LFQ"]


def hard_sign(logits):
    return torch.where(logits > 0, torch.ones_like(logits), -torch.ones_like(logits))


def test_joint_entropy_distinguishes_two_codes_from_full_codebook():
    lfq_class = load_lfq_class()
    model = lfq_class(in_dim=11, bits=11)

    full_logits = ((2 * model.code_bits - 1) * 20).unsqueeze(-1)
    full_terms = model._loss_terms(full_logits, hard_sign(full_logits))

    two_code_logits = torch.stack(
        [torch.full((11, 1), -20.0), torch.full((11, 1), 20.0)]
    )
    two_code_terms = model._loss_terms(two_code_logits, hard_sign(two_code_logits))

    assert float(full_terms["entropy"]) == pytest.approx(-math.log(2048), abs=2e-5)
    assert float(two_code_terms["entropy"]) == pytest.approx(-math.log(2), abs=2e-5)
    assert float(full_terms["joint_probability_sum"]) == pytest.approx(1.0, abs=1e-5)
    assert float(two_code_terms["joint_probability_sum"]) == pytest.approx(1.0, abs=1e-5)
    assert float(full_terms["entropy"]) < float(two_code_terms["entropy"]) - 6.0


def test_commitment_penalizes_distance_from_binary_codes():
    lfq_class = load_lfq_class()
    model = lfq_class(in_dim=3, bits=3)
    hard = torch.tensor([[[1.0], [-1.0], [1.0]]])

    at_code = model._loss_terms(hard, hard)
    scaled = model._loss_terms(3 * hard, hard)

    assert float(at_code["commitment"]) == 0.0
    assert float(scaled["commitment"]) == pytest.approx(4.0, abs=1e-6)


def test_forward_shapes_indices_and_gradients_are_valid():
    torch.manual_seed(0)
    lfq_class = load_lfq_class()
    model = lfq_class(in_dim=64, bits=11)
    z = torch.randn(2, 64, 25, requires_grad=True)

    zq, aux, indices, extra = model(z)
    aux.backward()

    assert zq.shape == z.shape
    assert indices.shape == (2, 25)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 2048
    assert extra["bit_probs"].shape == (2, 11, 25)
    assert float(extra["joint_probability_sum"]) == pytest.approx(1.0, abs=1e-5)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_joint_entropy_matches_brute_force_and_has_its_own_gradient():
    torch.manual_seed(1)
    lfq_class = load_lfq_class()
    model = lfq_class(in_dim=3, bits=3, temperature=0.7)
    logits = torch.randn(2, 3, 4, requires_grad=True)
    hard = hard_sign(logits)

    terms = model._loss_terms(logits, hard)

    probs = torch.sigmoid((logits / model.temperature).permute(0, 2, 1))
    flat_probs = probs.reshape(-1, model.bits)
    code_bits = model.code_bits.bool()
    joint = torch.where(
        code_bits.unsqueeze(0),
        flat_probs.unsqueeze(1),
        1 - flat_probs.unsqueeze(1),
    ).prod(-1)
    mean_joint = joint.mean(0)
    brute_force_entropy = -(mean_joint * mean_joint.log()).sum()

    assert float(terms["codebook_entropy"].detach()) == pytest.approx(
        float(brute_force_entropy.detach()), abs=1e-6
    )

    entropy_grad = torch.autograd.grad(terms["entropy"], logits)[0]
    assert torch.isfinite(entropy_grad).all()
    assert float(entropy_grad.norm()) > 0
