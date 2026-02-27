from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Mapping

import torch
from omegaconf import DictConfig


def build_adamw_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
    *,
    section: str,
    default_lr: float,
    default_weight_decay: float,
    betas_key: str = "betas",
    lr_key: str = "lr",
    weight_decay_key: str = "weight_decay",
    eps_key: str = "eps",
) -> torch.optim.Optimizer:
    train_cfg = cfg[section]
    lr = float(train_cfg.get(lr_key, default_lr))
    weight_decay = float(train_cfg.get(weight_decay_key, default_weight_decay))

    betas_cfg = train_cfg.get(betas_key, [0.9, 0.95])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])
    eps = float(train_cfg.get(eps_key, 1e-8))

    kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "eps": eps,
    }

    params = inspect.signature(torch.optim.AdamW).parameters
    use_fused = "fused" in params and device.type == "cuda"
    use_foreach = "foreach" in params and not use_fused

    if "fused" in params:
        kwargs["fused"] = use_fused
    if "foreach" in params:
        kwargs["foreach"] = use_foreach

    return torch.optim.AdamW(model.parameters(), **kwargs)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    *,
    section: str,
    max_steps_key: str = "max_steps",
    warmup_steps_key: str = "warmup_steps",
    min_lr_ratio_key: str = "min_lr_ratio",
    default_max_steps: int = 1000,
    default_warmup_steps: int = 100,
    default_min_lr_ratio: float = 0.1,
    step_delay_by_group_name: Mapping[str, int] | None = None,
    group_name_key: str = "group_name",
) -> torch.optim.lr_scheduler.LambdaLR:
    section_cfg = cfg[section]
    max_steps = max(1, int(section_cfg.get(max_steps_key, default_max_steps)))
    warmup_steps = max(0, int(section_cfg.get(warmup_steps_key, default_warmup_steps)))
    min_lr_ratio = float(section_cfg.get(min_lr_ratio_key, default_min_lr_ratio))

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        progress = (step_idx - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if not step_delay_by_group_name:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    group_lambdas: list[Callable[[int], float]] = []
    for param_group in optimizer.param_groups:
        group_name = str(param_group.get(group_name_key, ""))
        delay_steps = max(0, int(step_delay_by_group_name.get(group_name, 0)))
        if delay_steps <= 0:
            group_lambdas.append(lr_lambda)
            continue

        def delayed_lr_lambda(step_idx: int, *, _delay_steps: int = delay_steps) -> float:
            return lr_lambda(max(0, int(step_idx) - _delay_steps))

        group_lambdas.append(delayed_lr_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=group_lambdas)
