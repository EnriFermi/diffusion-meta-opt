from MetaOpt.env import episode_rollout


def eval_policy(policy, cfg, device, episode_sampler, episodes, model=None):
    returns = []
    val_scores = []

    for _ in range(episodes):
        ret, val_loss = episode_rollout(
            policy, cfg, device, episode_sampler, replay=None, training=False, model=model
        )
        returns.append(ret)
        val_scores.append(-val_loss)

    def mean_std(vals):
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        return mu, sd

    mu_r, sd_r = mean_std(returns)
    mu_v, sd_v = mean_std(val_scores)

    print(f"Eval meta-optimizer ({episodes} tasks): "
          f"mean reward={mu_r:.4f} std={sd_r:.4f} | "
          f"mean -val_loss={mu_v:.4f} std={sd_v:.4f}")
