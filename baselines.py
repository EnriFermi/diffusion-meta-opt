import torch
import torch.nn.functional as F

from MetaOpt.downstream import build_downstream_model, build_loss_fn
from MetaOpt.perf import (
    autocast_context,
    make_grad_scaler,
    maybe_channels_last_,
    maybe_compile,
    reset_parameters_,
)
from MetaOpt.utils import get_params


def _baseline_episode(optim_name, cfg, device, episode_sampler, model):
    maybe_channels_last_(model, enabled=bool(cfg.perf.channels_last) and str(cfg.downstream.name) == "cifar10")
    reset_parameters_(model)
    model.train()

    loss_fn = build_loss_fn(cfg)
    params = get_params(model)

    if optim_name == "adam":
        opt = torch.optim.Adam(params, lr=cfg.baseline.adam_lr)
    else:
        opt = torch.optim.SGD(params, lr=cfg.baseline.sgd_lr)

    amp_ctx = autocast_context(cfg, device)
    scaler = make_grad_scaler(cfg, device)

    batch = episode_sampler.sample(device)
    x_train, y_train, x_val, y_val = batch.x_train, batch.y_train, batch.x_val, batch.y_val
    if bool(cfg.perf.channels_last) and str(cfg.downstream.name) == "cifar10" and x_train.is_cuda:
        x_train = x_train.contiguous(memory_format=torch.channels_last)
        x_val = x_val.contiguous(memory_format=torch.channels_last)

    total_return = 0.0
    final_val_loss = None

    for step in range(cfg.meta.inner_steps):
        with amp_ctx:
            out = model(x_train)
            loss = loss_fn(out, y_train)
        reward = -loss.detach().item()

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        done = (step == cfg.meta.inner_steps - 1)
        if done:
            with torch.no_grad():
                with amp_ctx:
                    val_out = model(x_val)
                    final_val_loss = loss_fn(val_out, y_val).float().item()
            reward += -final_val_loss

        total_return += reward

    return total_return, final_val_loss


def run_baselines(cfg, device, episode_sampler):
    def mean_std(vals):
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        return mu, sd

    adam_rets, adam_vals = [], []
    sgd_rets, sgd_vals = [], []

    model = build_downstream_model(cfg).to(device)
    model = maybe_compile(model, cfg)

    for _ in range(cfg.baseline.episodes):
        r, v = _baseline_episode("adam", cfg, device, episode_sampler, model)
        adam_rets.append(r)
        adam_vals.append(-v)

        r, v = _baseline_episode("sgd", cfg, device, episode_sampler, model)
        sgd_rets.append(r)
        sgd_vals.append(-v)

    mu_ar, sd_ar = mean_std(adam_rets)
    mu_av, sd_av = mean_std(adam_vals)
    mu_sr, sd_sr = mean_std(sgd_rets)
    mu_sv, sd_sv = mean_std(sgd_vals)

    print(f"Adam baseline ({cfg.baseline.episodes} tasks): "
          f"mean reward={mu_ar:.4f} std={sd_ar:.4f} | "
          f"mean -val_loss={mu_av:.4f} std={sd_av:.4f}")
    print(f"SGD  baseline ({cfg.baseline.episodes} tasks): "
          f"mean reward={mu_sr:.4f} std={sd_sr:.4f} | "
          f"mean -val_loss={mu_sv:.4f} std={sd_sv:.4f}")
    print("(reward = sum_t -train_loss_t - final_val_loss)")
