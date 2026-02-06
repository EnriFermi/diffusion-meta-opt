import time

import torch

from MetaOpt.baselines import run_baselines
from MetaOpt.checkpoint import save_checkpoint
from MetaOpt.config import EVAL_EVERY, get_cfg
from MetaOpt.downstream import build_downstream_model, build_episode_sampler
from MetaOpt.env import episode_rollout
from MetaOpt.eval import eval_policy
from MetaOpt.meta_algos import build_meta_agent, resolve_meta_algorithm
from MetaOpt.perf import configure_torch, maybe_channels_last_, maybe_compile
from MetaOpt.replay import ReplayBuffer
from MetaOpt.utils import get_params, set_seed


def main():
    cfg = get_cfg()
    device = torch.device(cfg.device)
    configure_torch(cfg)
    set_seed(cfg.seed)

    episode_sampler = build_episode_sampler(cfg)
    # CIFAR-10 sampler is expensive to construct (downloads/loads dataset),
    # so reuse it for baseline/eval to avoid triple-loading.
    if str(cfg.downstream.name) == "cifar10":
        baseline_sampler = episode_sampler
        eval_sampler = episode_sampler
    else:
        baseline_sampler = build_episode_sampler(cfg)
        eval_sampler = build_episode_sampler(cfg)

    downstream_name = str(cfg.downstream.name)
    task_count = None
    if downstream_name == "regression" and hasattr(cfg, "tasks") and cfg.tasks is not None:
        task_count = len(cfg.tasks.mix)
    print(
        f"Device={device} | inner_steps={cfg.meta.inner_steps} | "
        f"downstream={downstream_name}"
        + (f" | tasks={task_count}" if task_count is not None else "")
        + f" | meta={resolve_meta_algorithm(cfg)}"
    )

    run_baselines(cfg, device, baseline_sampler)

    dummy = build_downstream_model(cfg).to(device)
    action_dim = len(get_params(dummy))
    state_dim = 4 * action_dim

    meta_algo = resolve_meta_algorithm(cfg)
    agent = build_meta_agent(meta_algo, state_dim, action_dim, cfg, device)
    agent.actor = maybe_compile(agent.actor, cfg)
    if meta_algo == "sac":
        agent.critic1 = maybe_compile(agent.critic1, cfg)
        agent.critic2 = maybe_compile(agent.critic2, cfg)
        agent.target1 = maybe_compile(agent.target1, cfg)
        agent.target2 = maybe_compile(agent.target2, cfg)

    replay = ReplayBuffer(cfg.rl.buffer_size) if meta_algo == "sac" else None

    rollout_model = build_downstream_model(cfg).to(device)
    maybe_channels_last_(rollout_model, enabled=bool(cfg.perf.channels_last) and str(cfg.downstream.name) == "cifar10")
    rollout_model = maybe_compile(rollout_model, cfg)

    global_step = 0
    t0 = time.time()

    for ep in range(1, cfg.rl.train_episodes + 1):
        if meta_algo == "sac":
            ret, val_loss = episode_rollout(
                agent, cfg, device, episode_sampler, replay=replay, training=True, model=rollout_model
            )
        elif meta_algo == "velo":
            metrics = agent.update(
                cfg,
                eval_fn=lambda: episode_rollout(
                    agent, cfg, device, episode_sampler, replay=None, training=False, model=rollout_model
                )[0],
            )
            ret, val_loss = episode_rollout(
                agent, cfg, device, episode_sampler, replay=None, training=False, model=rollout_model
            )
        else:
            raise ValueError(f"Unknown meta algorithm: {meta_algo!r}")
        global_step += cfg.meta.inner_steps

        if meta_algo == "sac":
            if global_step > cfg.rl.warmup_steps:
                for _ in range(cfg.meta.inner_steps * cfg.rl.updates_per_step):
                    assert replay is not None
                    agent.update(replay, cfg)

        if ep == 1 or ep % 10 == 0:
            elapsed = time.time() - t0
            if meta_algo == "sac":
                alpha_val = agent.alpha_value() if hasattr(agent, "alpha_value") else None
                if alpha_val is None:
                    print(
                        f"ep {ep:4d} | reward={ret:.4f} | -val_loss={-val_loss:.4f} | "
                        f"buffer={len(replay) if replay is not None else 0} | {elapsed:.1f}s"
                    )
                else:
                    print(
                        f"ep {ep:4d} | reward={ret:.4f} | -val_loss={-val_loss:.4f} | "
                        f"buffer={len(replay) if replay is not None else 0} | alpha={alpha_val:.4f} | {elapsed:.1f}s"
                    )
            else:
                print(
                    f"ep {ep:4d} | reward={ret:.4f} | -val_loss={-val_loss:.4f} | "
                    f"diff_mean={metrics.mean_diff:.4f} | diff_std={metrics.std_diff:.4f} | {elapsed:.1f}s"
                )

        if ep % cfg.checkpoint.save_every_episodes == 0:
            save_checkpoint(cfg.checkpoint.path, cfg, agent, global_step, ep)
            print(f"Checkpoint saved to {cfg.checkpoint.path}")

        if ep % EVAL_EVERY == 0:
            eval_policy(agent, cfg, device, eval_sampler, episodes=cfg.baseline.episodes, model=rollout_model)

    save_checkpoint(cfg.checkpoint.path, cfg, agent, global_step, cfg.rl.train_episodes)
    print(f"Final checkpoint saved to {cfg.checkpoint.path}")

    eval_policy(agent, cfg, device, eval_sampler, episodes=cfg.baseline.episodes, model=rollout_model)


if __name__ == "__main__":
    main()
