import torch
from omegaconf import OmegaConf

from MetaOpt.downstream import build_downstream_model
from MetaOpt.meta_algos import build_meta_agent, resolve_meta_algorithm
from MetaOpt.utils import get_params


def save_checkpoint(path, cfg, agent, global_step, episode):
    meta_algo = resolve_meta_algorithm(cfg)
    obj = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        # Backward compat: keep "algo", but prefer "meta_algo" going forward.
        "algo": meta_algo,
        "meta_algo": meta_algo,
        "agent": agent.state_dict(),
        "global_step": global_step,
        "episode": episode,
    }
    torch.save(obj, path)


def load_agent_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    loaded_cfg = OmegaConf.create(ckpt["cfg"])

    if not hasattr(loaded_cfg, "downstream") or loaded_cfg.downstream is None:
        loaded_cfg.downstream = OmegaConf.create({"name": "regression"})
    if not hasattr(loaded_cfg, "meta") or loaded_cfg.meta is None:
        loaded_cfg.meta = OmegaConf.create({})

    dummy = build_downstream_model(loaded_cfg).to(device)
    action_dim = len(get_params(dummy))
    state_dim = 4 * action_dim

    algo = ckpt.get("meta_algo") or ckpt.get("algo") or resolve_meta_algorithm(loaded_cfg)
    loaded_cfg.meta.algorithm = str(algo)
    agent = build_meta_agent(str(algo), state_dim, action_dim, loaded_cfg, device)
    if "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
    else:
        if str(algo) != "sac":
            raise ValueError("Legacy checkpoints without 'agent' are only supported for SAC")
        legacy_state = {
            "actor": ckpt["actor"],
            "critic1": ckpt["critic1"],
            "critic2": ckpt["critic2"],
            "target1": ckpt["target1"],
            "target2": ckpt["target2"],
            "log_alpha": ckpt["log_alpha"],
        }
        agent.load_state_dict(legacy_state)
    return loaded_cfg, agent


def load_actor_from_checkpoint(path, device):
    loaded_cfg, agent = load_agent_from_checkpoint(path, device)
    if not hasattr(agent, "actor"):
        raise ValueError("Loaded agent does not expose an actor")
    agent.actor.eval()
    return loaded_cfg, agent.actor
