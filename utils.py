import random

import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_params(params):
    return torch.cat([p.detach().view(-1) for p in params])


def flatten_grads(params):
    out = []
    for p in params:
        if p.grad is None:
            out.append(torch.zeros_like(p).view(-1))
        else:
            out.append(p.grad.detach().view(-1))
    return torch.cat(out)


@torch.no_grad()
def apply_update(params, delta_flat):
    i = 0
    for p in params:
        n = p.numel()
        p.add_(delta_flat[i:i + n].view_as(p))
        i += n


def get_params(model):
    return [p for p in model.parameters() if p.requires_grad]
