import torch
import torch.nn.functional as F

from MetaOpt.downstream import build_downstream_model, build_loss_fn
from MetaOpt.perf import autocast_context, make_grad_scaler, maybe_channels_last_, reset_parameters_
from MetaOpt.utils import get_params


def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((), device=x.device)
    x32 = x.float()
    return torch.sqrt(torch.mean(x32 * x32) + eps)


@torch.no_grad()
def _apply_blockwise_sgd(params, action, lr_scale: float = 1.0):
    """
    Apply a learned scalar step per parameter-tensor:
      p <- p - lr_scale * a_i * grad(p)
    """
    for i, p in enumerate(params):
        if p.grad is None:
            continue
        p.add_(-lr_scale * action[i] * p.grad)


def episode_rollout(policy, cfg, device, episode_sampler, replay=None, training=True, model=None):
    if model is None:
        model = build_downstream_model(cfg).to(device)
    maybe_channels_last_(model, enabled=bool(cfg.perf.channels_last) and str(cfg.downstream.name) == "cifar10")
    reset_parameters_(model)
    model.train()

    loss_fn = build_loss_fn(cfg)
    amp_ctx = autocast_context(cfg, device)
    scaler = make_grad_scaler(cfg, device)

    batch = episode_sampler.sample(device)
    x_train, y_train, x_val, y_val = batch.x_train, batch.y_train, batch.x_val, batch.y_val
    if bool(cfg.perf.channels_last) and str(cfg.downstream.name) == "cifar10" and x_train.is_cuda:
        x_train = x_train.contiguous(memory_format=torch.channels_last)
        x_val = x_val.contiguous(memory_format=torch.channels_last)

    params = get_params(model)
    n_blocks = len(params)
    if n_blocks == 0:
        raise ValueError("Downstream model has no trainable parameters")

    dummy_opt = torch.optim.SGD(params, lr=0.0) if scaler is not None else None

    m = torch.zeros(n_blocks, device=device)
    v = torch.zeros(n_blocks, device=device)

    # initial gradient and state (s_0)
    model.zero_grad(set_to_none=True)
    with amp_ctx:
        out0 = model(x_train)
        loss0 = loss_fn(out0, y_train)
    if scaler is not None:
        scaler.scale(loss0).backward()
        assert dummy_opt is not None
        scaler.unscale_(dummy_opt)
    else:
        loss0.backward()

    w_rms = torch.stack([_rms(p.detach()) for p in params])
    g_rms = torch.stack([_rms(p.grad.detach()) if p.grad is not None else torch.zeros((), device=device) for p in params])

    m = cfg.meta.mom_beta * m + (1.0 - cfg.meta.mom_beta) * g_rms
    v = cfg.meta.adam_beta2 * v + (1.0 - cfg.meta.adam_beta2) * (g_rms * g_rms)

    state = torch.cat([w_rms, g_rms, m, v], dim=0).detach()

    total_return = 0.0
    final_val_loss = None

    for step in range(cfg.meta.inner_steps):
        s = state.unsqueeze(0)

        with torch.no_grad():
            a = policy.select_action(s, deterministic=not training)
        a = a.squeeze(0)  # (n_blocks,)

        _apply_blockwise_sgd(params, a)

        model.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(x_train)
            loss = loss_fn(out, y_train)
        reward = -loss.detach().item()

        if scaler is not None:
            scaler.scale(loss).backward()
            assert dummy_opt is not None
            scaler.unscale_(dummy_opt)
        else:
            loss.backward()

        w_rms = torch.stack([_rms(p.detach()) for p in params])
        g_rms = torch.stack([_rms(p.grad.detach()) if p.grad is not None else torch.zeros((), device=device) for p in params])

        m = cfg.meta.mom_beta * m + (1.0 - cfg.meta.mom_beta) * g_rms
        v = cfg.meta.adam_beta2 * v + (1.0 - cfg.meta.adam_beta2) * (g_rms * g_rms)

        next_state = torch.cat([w_rms, g_rms, m, v], dim=0).detach()

        done = float(step == cfg.meta.inner_steps - 1)
        if done:
            with torch.no_grad():
                with amp_ctx:
                    val_out = model(x_val)
                    final_val_loss = loss_fn(val_out, y_val).float().item()
            # terminal shaping: add -val_loss to last step reward
            reward += -final_val_loss

        total_return += reward

        if replay is not None:
            replay.push(state.cpu(), a.cpu(), reward, next_state.cpu(), done)

        state = next_state

        if scaler is not None:
            scaler.update()

    return total_return, final_val_loss
