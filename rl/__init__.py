from MetaOpt.rl.sac import SACAgent


ALGO_REGISTRY = {
    "sac": SACAgent,
}


def build_agent(algo_name, state_dim, action_dim, cfg, device):
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown RL algorithm: {algo_name}")
    return ALGO_REGISTRY[algo_name](state_dim=state_dim, action_dim=action_dim, cfg=cfg, device=device)
