from __future__ import annotations

import torch

from MetaOpt.rl import build_agent as build_rl_agent
from MetaOpt.velo import VeloESAgent


def resolve_meta_algorithm(cfg) -> str:
    if hasattr(cfg, "meta") and cfg.meta is not None and cfg.meta.get("algorithm") is not None:
        return str(cfg.meta.algorithm)
    if hasattr(cfg, "rl") and cfg.rl is not None and cfg.rl.get("algorithm") is not None:
        return str(cfg.rl.algorithm)
    return "sac"


def build_meta_agent(algo_name: str, state_dim: int, action_dim: int, cfg, device: torch.device):
    algo_name = str(algo_name)
    if algo_name == "sac":
        return build_rl_agent("sac", state_dim, action_dim, cfg, device)
    if algo_name == "velo":
        return VeloESAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg, device=device)
    raise ValueError(f"Unknown meta algorithm: {algo_name!r}")

