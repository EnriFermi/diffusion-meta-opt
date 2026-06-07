from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Mapping

import torch
from omegaconf import DictConfig


_WRAPPER_PARAM_NAME_PREFIXES = ("module.", "_orig_mod.")
_PATCH_TOKENIZER_ALPHA_GROUP = "patch_tokenizer_alpha"


def _strip_optimizer_wrapper_prefixes(name: str) -> str:
    normalized = str(name)
    while True:
        for prefix in _WRAPPER_PARAM_NAME_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        else:
            return normalized


def _is_patch_tokenizer_alpha_parameter(name: str) -> bool:
    parts = _strip_optimizer_wrapper_prefixes(name).split(".")
    return (
        len(parts) == 4
        and parts[0] == "patch_tokenizer"
        and parts[1] == "blocks"
        and parts[3] == "alpha"
    )


def _build_patch_tokenizer_alpha_param_groups(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    alpha_lr: float,
    alpha_weight_decay: float,
) -> list[dict[str, Any]] | None:
    if alpha_lr <= 0.0:
        return None

    main_params: list[torch.nn.Parameter] = []
    alpha_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if _is_patch_tokenizer_alpha_parameter(name):
            alpha_params.append(param)
        else:
            main_params.append(param)

    if not alpha_params:
        return None

    param_groups: list[dict[str, Any]] = []
    if main_params:
        param_groups.append(
            {
                "params": main_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": "main",
            }
        )
    param_groups.append(
        {
            "params": alpha_params,
            "lr": alpha_lr,
            "weight_decay": alpha_weight_decay,
            "group_name": _PATCH_TOKENIZER_ALPHA_GROUP,
        }
    )
    return param_groups


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
    optimizer_name = str(train_cfg.get("optimizer_name", "adamw")).strip().lower()
    lr = float(train_cfg.get(lr_key, default_lr))
    weight_decay = float(train_cfg.get(weight_decay_key, default_weight_decay))
    patch_tokenizer_alpha_lr = float(train_cfg.get("patch_tokenizer_alpha_lr", 0.0) or 0.0)
    patch_tokenizer_alpha_weight_decay = float(
        train_cfg.get("patch_tokenizer_alpha_weight_decay", 0.0) or 0.0
    )

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

    if optimizer_name == "adamw":
        optimizer_cls = torch.optim.AdamW
    elif optimizer_name == "adam":
        optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(
            f"Unsupported {section}.optimizer_name={optimizer_name!r}. Expected one of: 'adamw', 'adam'"
        )

    params = inspect.signature(optimizer_cls).parameters
    use_fused = "fused" in params and device.type == "cuda"
    use_foreach = "foreach" in params and not use_fused

    if "fused" in params:
        kwargs["fused"] = use_fused
    if "foreach" in params:
        kwargs["foreach"] = use_foreach

    optimizer_params = _build_patch_tokenizer_alpha_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        alpha_lr=patch_tokenizer_alpha_lr,
        alpha_weight_decay=patch_tokenizer_alpha_weight_decay,
    )
    if optimizer_params is None:
        return optimizer_cls(model.parameters(), **kwargs)
    return optimizer_cls(optimizer_params, **kwargs)


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
) -> torch.optim.lr_scheduler.LambdaLR | None:
    section_cfg = cfg[section]
    scheduler_name = str(section_cfg.get("scheduler_name", "cosine")).strip().lower()
    if scheduler_name in {"none", "off", "false"}:
        return None
    if scheduler_name not in {"cosine", "cosine_decay"}:
        raise ValueError(
            f"Unsupported {section}.scheduler_name={scheduler_name!r}. Expected one of: 'cosine', 'none'"
        )
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
