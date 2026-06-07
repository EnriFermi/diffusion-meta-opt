from __future__ import annotations

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from training.optim import build_adamw_optimizer


class _PatchBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, 1, 4))
        self.proj = torch.nn.Linear(4, 4)


class _OtherAlpha(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(()))


class _ToyBigVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_tokenizer = torch.nn.Module()
        self.patch_tokenizer.blocks = torch.nn.ModuleList([_PatchBlock()])
        self.encoder_conditioning_adapters = torch.nn.ModuleList([_OtherAlpha()])
        self.head = torch.nn.Linear(4, 1)


class _CompiledDdpLikeWrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.module = torch.nn.Module()
        self.module._orig_mod = inner


def _param_ids(param_group: dict[str, object]) -> set[int]:
    return {id(param) for param in param_group["params"]}  # type: ignore[index]


def test_patch_tokenizer_alpha_lr_builds_dedicated_param_group_with_wrapped_names() -> None:
    model = _CompiledDdpLikeWrapper(_ToyBigVAE())
    cfg = OmegaConf.create(
        {
            "train": {
                "optimizer_name": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-2,
                "patch_tokenizer_alpha_lr": 5e-3,
                "patch_tokenizer_alpha_weight_decay": 0.0,
            }
        }
    )

    optimizer = build_adamw_optimizer(
        model=model,
        cfg=cfg,
        device=torch.device("cpu"),
        section="train",
        default_lr=3e-4,
        default_weight_decay=0.01,
    )

    assert [group.get("group_name") for group in optimizer.param_groups] == ["main", "patch_tokenizer_alpha"]
    assert [float(group["lr"]) for group in optimizer.param_groups] == [1e-3, 5e-3]
    assert [float(group["weight_decay"]) for group in optimizer.param_groups] == [1e-2, 0.0]

    alpha_param = model.module._orig_mod.patch_tokenizer.blocks[0].alpha
    other_alpha_param = model.module._orig_mod.encoder_conditioning_adapters[0].alpha
    assert id(alpha_param) in _param_ids(optimizer.param_groups[1])
    assert id(other_alpha_param) in _param_ids(optimizer.param_groups[0])


def test_patch_tokenizer_alpha_lr_disabled_keeps_single_legacy_param_group() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "optimizer_name": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-2,
                "patch_tokenizer_alpha_lr": 0.0,
            }
        }
    )

    optimizer = build_adamw_optimizer(
        model=_ToyBigVAE(),
        cfg=cfg,
        device=torch.device("cpu"),
        section="train",
        default_lr=3e-4,
        default_weight_decay=0.01,
    )

    assert len(optimizer.param_groups) == 1
    assert "group_name" not in optimizer.param_groups[0]
